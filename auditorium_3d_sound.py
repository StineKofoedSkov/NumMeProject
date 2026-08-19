# %%
"""3D FDTD-model af auditoriet med et WAV-klip som lydkilde.

Modellen er lavet som en overskuelig undervisningsmodel:

* geometrien og lydfeltet bruger det samme koordinatsystem
* vægge, tavle, loft, gulv og sæder ligger i et 3D-materialegitter
* materialernes absorption dæmper luftcellerne langs materialefladerne
* lydtrykket registreres ved hvert sæde
* Plotly viser både lokalet og animerede 3D-isoflader af lydtrykket

Absorptionsmodellen er en tilnærmelse. Frekvensafhængige impedansrandbetingelser
kan tilføjes senere uden at ændre geometrien eller visualiseringen.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
from scipy.signal import butter, filtfilt, welch


# %%
# ------------------------------------------------------------
# 1. Indstillinger
# ------------------------------------------------------------

C = 343.0

ROOM_LENGTH = 8.80
ROOM_WIDTH = 7.20
ROOM_HEIGHT = 2.92

PROFESSOR_AREA_LENGTH = 1.60
NUMBER_OF_ROWS = 8
SEATS_PER_ROW = 8
AISLE_WIDTH = 0.80

LOWEST_EAR_HEIGHT = 0.85
HIGHEST_EAR_HEIGHT = 1.22
FLOOR_RISE = HIGHEST_EAR_HEIGHT - LOWEST_EAR_HEIGHT

SEAT_WIDTH = 0.50
SEAT_DEPTH = 0.50
SEATED_EAR_HEIGHT_ABOVE_FLOOR = 1.10

PROFESSOR_POSITION = np.array([0.80, ROOM_WIDTH / 2, 1.65])

# En mindre værdi giver flere gitterpunkter, højere opløselig frekvens
# og væsentligt længere beregningstid.
GRID_SPACING = 0.18
POINTS_PER_WAVELENGTH = 10
CFL_SAFETY = 0.70

# Start med et kort udsnit. Forøg først når modellen fungerer på din computer.
SIMULATION_DURATION = 0.05
MAX_PLOTLY_FRAMES = 15
VISUALIZATION_SKIP = 2
SOURCE_AMPLITUDE = 0.50

AUDIO_PATH = Path("Stine_fixed.wav")


# %%
# ------------------------------------------------------------
# 2. Rumgeometri og sædeplaceringer
# ------------------------------------------------------------

row_depth = (ROOM_LENGTH - PROFESSOR_AREA_LENGTH) / NUMBER_OF_ROWS
first_row_x = PROFESSOR_AREA_LENGTH + 0.5 * row_depth
last_row_x = PROFESSOR_AREA_LENGTH + (NUMBER_OF_ROWS - 0.5) * row_depth

row_x_positions = np.linspace(first_row_x, last_row_x, NUMBER_OF_ROWS)
row_floor_heights = np.linspace(0.0, FLOOR_RISE, NUMBER_OF_ROWS)
seat_y_positions = np.linspace(
    AISLE_WIDTH + 0.35,
    ROOM_WIDTH - AISLE_WIDTH - 0.35,
    SEATS_PER_ROW,
)

seat_positions = np.array(
    [
        [row_x_positions[row], seat_y, row_floor_heights[row]]
        for row in range(NUMBER_OF_ROWS)
        for seat_y in seat_y_positions
    ]
)


def floor_height(x_value, y_value):
    """Fladt gulv foran/i gangene og jævn hældning under sæderne."""
    x_value, y_value = np.broadcast_arrays(
        np.asarray(x_value, dtype=float),
        np.asarray(y_value, dtype=float),
    )
    height = np.zeros_like(x_value)

    seating_width = (y_value >= AISLE_WIDTH) & (
        y_value <= ROOM_WIDTH - AISLE_WIDTH
    )
    behind_professor = x_value >= PROFESSOR_AREA_LENGTH
    seating_area = seating_width & behind_professor

    slope = (x_value - first_row_x) / (last_row_x - first_row_x)
    slope = np.clip(slope, 0.0, 1.0)
    height[seating_area] = FLOOR_RISE * slope[seating_area]
    return height


# %%
# ------------------------------------------------------------
# 3. WAV-fil: analyse, filtrering og sampling ved FDTD-tiderne
# ------------------------------------------------------------


def read_mono_audio(wav_path):
    audio, sample_rate = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    peak = np.max(np.abs(audio))
    if peak == 0:
        raise ValueError("Lydfilen indeholder kun nuller")
    return audio / peak, sample_rate


def analyse_audio(audio, sample_rate, resolved_frequency):
    frequencies, power = welch(
        audio,
        fs=sample_rate,
        nperseg=min(4096, len(audio)),
    )
    dominant_frequency = frequencies[np.argmax(power)]

    print(f"Lydfilens samplerate: {sample_rate} Hz")
    print(f"Lydfilens længde: {len(audio) / sample_rate:.2f} s")
    print(f"Dominerende frekvens: {dominant_frequency:.1f} Hz")
    print(f"Gitterets omtrentlige frekvensgrænse: {resolved_frequency:.1f} Hz")

    time = np.arange(len(audio)) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(time, audio, color="darkmagenta", linewidth=0.7)
    axes[0].set(xlabel="Tid [s]", ylabel="Amplitude", title="Lydklip")
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(frequencies, power, color="purple")
    axes[1].axvline(
        resolved_frequency,
        color="red",
        linestyle="--",
        label="Gitterets frekvensgrænse",
    )
    axes[1].set_xlim(0, min(3000, sample_rate / 2))
    axes[1].set(
        xlabel="Frekvens [Hz]",
        ylabel="Effekttæthed",
        title="Frekvensspektrum",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def make_source_signal(audio, sample_rate, dt, number_of_steps, f_max):
    """Lav præcis én kildeværdi for hvert FDTD-tidstrin."""
    nyquist = sample_rate / 2
    cutoff_hz = min(0.90 * f_max, 0.95 * nyquist)
    normalized_cutoff = cutoff_hz / nyquist

    if normalized_cutoff < 0.99:
        b, a = butter(4, normalized_cutoff, btype="low")
        filtered = filtfilt(b, a, audio)
    else:
        filtered = audio

    simulation_times = np.arange(number_of_steps) * dt
    audio_times = np.arange(len(filtered)) / sample_rate
    signal = np.interp(simulation_times, audio_times, filtered, left=0.0, right=0.0)

    # Kort indfasning begrænser et kunstigt klik ved t = 0.
    ramp_length = min(100, len(signal))
    signal[:ramp_length] *= np.linspace(0.0, 1.0, ramp_length)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = SOURCE_AMPLITUDE * signal / peak
    return signal.astype(np.float32)


# %%
# ------------------------------------------------------------
# 4. 3D-gitter og materialer
# ------------------------------------------------------------

# Arrayrækkefølgen er altid [z, y, x].
Nx = int(np.ceil(ROOM_LENGTH / GRID_SPACING)) + 1
Ny = int(np.ceil(ROOM_WIDTH / GRID_SPACING)) + 1
Nz = int(np.ceil(ROOM_HEIGHT / GRID_SPACING)) + 1

x = np.linspace(0.0, ROOM_LENGTH, Nx)
y = np.linspace(0.0, ROOM_WIDTH, Ny)
z = np.linspace(0.0, ROOM_HEIGHT, Nz)

dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

dt_limit = 1.0 / (C * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))
dt = CFL_SAFETY * dt_limit
Nt = int(np.ceil(SIMULATION_DURATION / dt))
simulation_times = np.arange(Nt) * dt
f_max = C / (POINTS_PER_WAVELENGTH * max(dx, dy, dz))

print(f"Gitter: {Nx} x {Ny} x {Nz} = {Nx * Ny * Nz:,} celler")
print(f"dx={dx:.3f} m, dy={dy:.3f} m, dz={dz:.3f} m")
print(f"dt={dt:.2e} s, Nt={Nt}, varighed={Nt * dt:.3f} s")
print(f"Maksimal omtrentligt opløselig frekvens: {f_max:.0f} Hz")

AIR = 0
BRICK = 1
WINDOW = 2
BOARD = 3
WOOD = 4
CEILING = 5
SEAT = 6

MATERIAL_NAMES = {
    AIR: "Luft",
    BRICK: "Mursten",
    WINDOW: "Vindue",
    BOARD: "Tavle",
    WOOD: "Trægulv",
    CEILING: "Bræddeloft",
    SEAT: "Polstret sæde",
}

# Foreløbige bredbåndsværdier. Senere kan de erstattes med én værdi pr. frekvensbånd.
ABSORPTION = np.array(
    [
        0.00,  # luft
        0.03,  # mursten
        0.08,  # vindue
        0.10,  # tavle
        0.12,  # træ
        0.25,  # bræddeloft
        0.65,  # polstret sæde
    ],
    dtype=np.float32,
)

material = np.full((Nz, Ny, Nx), AIR, dtype=np.uint8)


def coordinate_slice(axis, lower, upper):
    """Indekser, hvis koordinater ligger i et lukket interval."""
    indices = np.flatnonzero((axis >= lower) & (axis <= upper))
    if len(indices) == 0:
        index = int(np.argmin(np.abs(axis - 0.5 * (lower + upper))))
        return slice(index, index + 1)
    return slice(indices[0], indices[-1] + 1)


def voxel_box(center_x, center_y, bottom_z, size_x, size_y, size_z, material_id):
    xs = coordinate_slice(x, center_x - size_x / 2, center_x + size_x / 2)
    ys = coordinate_slice(y, center_y - size_y / 2, center_y + size_y / 2)
    zs = coordinate_slice(z, bottom_z, bottom_z + size_z)
    material[zs, ys, xs] = material_id


# Ydervægge. Tavlen overskriver senere en del af forvæggen.
material[:, :, 0] = BRICK
material[:, :, -1] = BRICK
material[:, 0, :] = WINDOW
material[:, -1, :] = BRICK
material[-1, :, :] = CEILING

# Trægulv, inklusive det hældende område.
Grid_X, Grid_Y = np.meshgrid(x, y, indexing="xy")
Grid_Floor_Z = floor_height(Grid_X, Grid_Y)
for iy in range(Ny):
    for ix in range(Nx):
        top_floor_index = int(np.searchsorted(z, Grid_Floor_Z[iy, ix], side="right") - 1)
        material[: top_floor_index + 1, iy, ix] = WOOD

# Tavle: 40 cm væg over og under samt 40 cm væg til højre.
board_y_slice = coordinate_slice(y, 0.0, ROOM_WIDTH - 0.40)
board_z_slice = coordinate_slice(z, 0.40, ROOM_HEIGHT - 0.40)
material[board_z_slice, board_y_slice, 0] = BOARD

# Polstrede sæder i samme koordinater som Plotly-modellen.
for seat_x, seat_y, seat_floor_z in seat_positions:
    voxel_box(seat_x, seat_y, seat_floor_z + 0.32, 0.45, 0.50, 0.12, SEAT)
    voxel_box(seat_x + 0.18, seat_y, seat_floor_z + 0.40, 0.10, 0.50, 0.45, SEAT)

air = material == AIR


def build_surface_absorption(material_grid):
    """Absorption i luftceller, der rører ved en materialeflade."""
    result = np.zeros_like(material_grid, dtype=np.float32)

    directions = [
        ((slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))),
        ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
        ((slice(None), slice(1, None), slice(None)), (slice(None), slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(1, None), slice(None), slice(None)), (slice(None, -1), slice(None), slice(None))),
        ((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
    ]

    for air_slice, neighbor_slice in directions:
        air_part = material_grid[air_slice] == AIR
        neighbor_material = material_grid[neighbor_slice]
        touches_material = air_part & (neighbor_material != AIR)
        candidate = ABSORPTION[neighbor_material]
        result[air_slice] = np.maximum(
            result[air_slice],
            np.where(touches_material, candidate, 0.0),
        )
    return result


surface_absorption = build_surface_absorption(material)


# %%
# ------------------------------------------------------------
# 5. Indlæs lyd og kør den tredimensionelle FDTD-simulering
# ------------------------------------------------------------

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Kan ikke finde {AUDIO_PATH}. Læg WAV-filen ved siden af dette script "
        "eller ret AUDIO_PATH øverst i filen"
    )

audio, sample_rate = read_mono_audio(AUDIO_PATH)
analyse_audio(audio, sample_rate, f_max)
source_signal = make_source_signal(audio, sample_rate, dt, Nt, f_max)


def nearest_index(axis, coordinate):
    return int(np.argmin(np.abs(axis - coordinate)))


source_ix = nearest_index(x, PROFESSOR_POSITION[0])
source_iy = nearest_index(y, PROFESSOR_POSITION[1])
source_iz = nearest_index(z, PROFESSOR_POSITION[2])

if not air[source_iz, source_iy, source_ix]:
    raise ValueError("Lydkilden er placeret inde i et materiale")

seat_ear_positions = seat_positions.copy()
seat_ear_positions[:, 2] += SEATED_EAR_HEIGHT_ABOVE_FLOOR

seat_ix = np.array([nearest_index(x, value) for value in seat_ear_positions[:, 0]])
seat_iy = np.array([nearest_index(y, value) for value in seat_ear_positions[:, 1]])
seat_iz = np.array([nearest_index(z, value) for value in seat_ear_positions[:, 2]])

u_old = np.zeros((Nz, Ny, Nx), dtype=np.float32)
u = np.zeros_like(u_old)
u_new = np.zeros_like(u_old)

cx2 = np.float32((C * dt / dx) ** 2)
cy2 = np.float32((C * dt / dy) ** 2)
cz2 = np.float32((C * dt / dz) ** 2)

interior_air = air[1:-1, 1:-1, 1:-1]
loss = surface_absorption[1:-1, 1:-1, 1:-1]

save_every = max(1, Nt // MAX_PLOTLY_FRAMES)
volume_frames = []
frame_times = []
seat_energy = np.zeros(len(seat_positions), dtype=np.float64)

print("Kører 3D-simuleringen ...")

for step in range(Nt):
    center = u[1:-1, 1:-1, 1:-1]

    # Ved et fast materiale erstattes naboværdien med centerets værdi.
    # Det svarer til en omtrentligt reflekterende nul-normalgradient.
    xp = np.where(air[1:-1, 1:-1, 2:], u[1:-1, 1:-1, 2:], center)
    xm = np.where(air[1:-1, 1:-1, :-2], u[1:-1, 1:-1, :-2], center)
    yp = np.where(air[1:-1, 2:, 1:-1], u[1:-1, 2:, 1:-1], center)
    ym = np.where(air[1:-1, :-2, 1:-1], u[1:-1, :-2, 1:-1], center)
    zp = np.where(air[2:, 1:-1, 1:-1], u[2:, 1:-1, 1:-1], center)
    zm = np.where(air[:-2, 1:-1, 1:-1], u[:-2, 1:-1, 1:-1], center)

    laplace_term = (
        cx2 * (xp - 2.0 * center + xm)
        + cy2 * (yp - 2.0 * center + ym)
        + cz2 * (zp - 2.0 * center + zm)
    )

    previous = u_old[1:-1, 1:-1, 1:-1]

    # Diskretiseret dæmpet bølgeligning ved materialefladerne.
    # loss=0 giver den almindelige centrale tidsdifferens.
    calculated = (
        (2.0 - loss) * center
        - (1.0 - loss) * previous
        + laplace_term
    )

    u_new.fill(0.0)
    u_new[1:-1, 1:-1, 1:-1] = np.where(interior_air, calculated, 0.0)
    u_new[source_iz, source_iy, source_ix] += source_signal[step]

    seat_pressure = u_new[seat_iz, seat_iy, seat_ix]
    seat_energy += seat_pressure.astype(np.float64) ** 2

    if step % save_every == 0 or step == Nt - 1:
        volume_frames.append(
            u_new[::VISUALIZATION_SKIP, ::VISUALIZATION_SKIP, ::VISUALIZATION_SKIP].copy()
        )
        frame_times.append(step * dt)

    u_old, u, u_new = u, u_new, u_old

volume_frames = np.asarray(volume_frames, dtype=np.float32)
frame_times = np.asarray(frame_times)

seat_rms = np.sqrt(seat_energy / Nt)
seat_relative_db = 20.0 * np.log10(seat_rms / (np.max(seat_rms) + 1e-12) + 1e-12)

print(f"Gemte {len(volume_frames)} Plotly-frames")
print(f"Sædeniveauer: {seat_relative_db.min():.1f} til {seat_relative_db.max():.1f} dB relativt")


# %%
# ------------------------------------------------------------
# 6. Plotly-geometri
# ------------------------------------------------------------


def constant_surface(figure, x_values, y_values, z_values, color, opacity, name):
    figure.add_trace(
        go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            surfacecolor=np.zeros_like(np.asarray(z_values), dtype=float),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=name,
            showlegend=True,
            hoverinfo="name",
        )
    )


def add_box_mesh(vertices, triangles_i, triangles_j, triangles_k, center_x, center_y,
                 bottom_z, size_x, size_y, size_z):
    start = len(vertices)
    x_min, x_max = center_x - size_x / 2, center_x + size_x / 2
    y_min, y_max = center_y - size_y / 2, center_y + size_y / 2
    z_min, z_max = bottom_z, bottom_z + size_z

    vertices.extend(
        [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]
    )
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    for i, j, k in faces:
        triangles_i.append(start + i)
        triangles_j.append(start + j)
        triangles_k.append(start + k)


def make_room_figure():
    figure = go.Figure()

    floor_plot_x = np.linspace(0, ROOM_LENGTH, 120)
    floor_plot_y = np.linspace(0, ROOM_WIDTH, 100)
    floor_x_grid, floor_y_grid = np.meshgrid(floor_plot_x, floor_plot_y)
    floor_z_grid = floor_height(floor_x_grid, floor_y_grid)
    constant_surface(
        figure, floor_x_grid, floor_y_grid, floor_z_grid,
        "rgb(166,112,66)", 0.90, "Trægulv"
    )

    wall_y, wall_z = np.meshgrid(
        np.linspace(0, ROOM_WIDTH, 40), np.linspace(0, ROOM_HEIGHT, 30)
    )
    constant_surface(
        figure, np.zeros_like(wall_y), wall_y, wall_z,
        "rgb(150,85,65)", 0.25, "Forvæg"
    )
    constant_surface(
        figure, np.full_like(wall_y, ROOM_LENGTH), wall_y, wall_z,
        "rgb(150,85,65)", 0.18, "Bagvæg"
    )

    side_x, side_z = np.meshgrid(
        np.linspace(0, ROOM_LENGTH, 50), np.linspace(0, ROOM_HEIGHT, 25)
    )
    constant_surface(
        figure, side_x, np.zeros_like(side_x), side_z,
        "rgb(120,190,220)", 0.18, "Vinduesvæg"
    )
    constant_surface(
        figure, side_x, np.full_like(side_x, ROOM_WIDTH), side_z,
        "rgb(150,85,65)", 0.18, "Murstensvæg"
    )

    ceiling_x, ceiling_y = np.meshgrid(
        np.linspace(0, ROOM_LENGTH, 50), np.linspace(0, ROOM_WIDTH, 40)
    )
    constant_surface(
        figure, ceiling_x, ceiling_y, np.full_like(ceiling_x, ROOM_HEIGHT),
        "rgb(188,145,95)", 0.08, "Bræddeloft"
    )

    board_y, board_z = np.meshgrid(
        np.linspace(0, ROOM_WIDTH - 0.40, 40),
        np.linspace(0.40, ROOM_HEIGHT - 0.40, 25),
    )
    constant_surface(
        figure, np.full_like(board_y, 0.005), board_y, board_z,
        "rgb(25,75,52)", 1.0, "Tavle"
    )

    vertices, triangles_i, triangles_j, triangles_k = [], [], [], []
    for seat_x_value, seat_y_value, floor_z_value in seat_positions:
        add_box_mesh(
            vertices, triangles_i, triangles_j, triangles_k,
            seat_x_value, seat_y_value, floor_z_value + 0.32,
            0.45, 0.50, 0.12,
        )
        add_box_mesh(
            vertices, triangles_i, triangles_j, triangles_k,
            seat_x_value + 0.18, seat_y_value, floor_z_value + 0.40,
            0.10, 0.50, 0.45,
        )

    vertices_array = np.asarray(vertices)
    figure.add_trace(
        go.Mesh3d(
            x=vertices_array[:, 0], y=vertices_array[:, 1], z=vertices_array[:, 2],
            i=triangles_i, j=triangles_j, k=triangles_k,
            color="rgb(180,35,45)", opacity=0.90, flatshading=True,
            name="Polstrede sæder", showlegend=True,
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=[PROFESSOR_POSITION[0]], y=[PROFESSOR_POSITION[1]],
            z=[PROFESSOR_POSITION[2]], mode="markers+text",
            marker=dict(size=8, color="lime", symbol="circle"),
            text=["Mathias"], textposition="top center", name="Lydkilde",
        )
    )

    # Målepunkterne viser samlet RMS-niveau ved sæderne, relativt til bedste sæde.
    figure.add_trace(
        go.Scatter3d(
            x=seat_ear_positions[:, 0],
            y=seat_ear_positions[:, 1],
            z=seat_ear_positions[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=seat_relative_db,
                colorscale="Turbo",
                cmin=-30,
                cmax=0,
                colorbar=dict(title="Sædeniveau<br>[relativ dB]", x=1.02),
            ),
            customdata=np.column_stack(
                [
                    np.repeat(np.arange(1, NUMBER_OF_ROWS + 1), SEATS_PER_ROW),
                    np.tile(np.arange(1, SEATS_PER_ROW + 1), NUMBER_OF_ROWS),
                    seat_relative_db,
                ]
            ),
            hovertemplate=(
                "Række %{customdata[0]:.0f}<br>"
                "Sæde %{customdata[1]:.0f}<br>"
                "%{customdata[2]:.1f} dB relativt<extra></extra>"
            ),
            name="Lydniveau ved sæder",
        )
    )

    figure.update_layout(
        scene=dict(
            xaxis=dict(title="Længde [m]", range=[0, ROOM_LENGTH]),
            yaxis=dict(title="Bredde [m]", range=[0, ROOM_WIDTH]),
            zaxis=dict(title="Højde [m]", range=[0, ROOM_HEIGHT]),
            aspectmode="manual",
            aspectratio=dict(
                x=ROOM_LENGTH / ROOM_WIDTH,
                y=1,
                z=ROOM_HEIGHT / ROOM_WIDTH,
            ),
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.1)),
        ),
        width=1100,
        height=750,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    return figure


# %%
# ------------------------------------------------------------
# 7. Animeret 3D-lydfelt i lokalet
# ------------------------------------------------------------

plot_x = x[::VISUALIZATION_SKIP]
plot_y = y[::VISUALIZATION_SKIP]
plot_z = z[::VISUALIZATION_SKIP]
Plot_Z, Plot_Y, Plot_X = np.meshgrid(plot_z, plot_y, plot_x, indexing="ij")

flat_x = Plot_X.ravel()
flat_y = Plot_Y.ravel()
flat_z = Plot_Z.ravel()

global_limit = np.percentile(np.abs(volume_frames), 99.5)
if global_limit <= 0:
    global_limit = 1.0


def sound_isosurface(frame, show_scale):
    magnitude = np.abs(frame).ravel()
    return go.Isosurface(
        x=flat_x,
        y=flat_y,
        z=flat_z,
        value=magnitude,
        isomin=0.25 * global_limit,
        isomax=global_limit,
        surface_count=3,
        colorscale=[
            [0.0, "rgb(230,220,255)"],
            [0.4, "rgb(160,105,220)"],
            [1.0, "rgb(70,0,120)"],
        ],
        opacity=0.22,
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=show_scale,
        colorbar=dict(title="|Lydtryk|", x=1.14) if show_scale else None,
        name="Lydfelt",
        hoverinfo="skip",
    )


room_figure = make_room_figure()
sound_trace_index = len(room_figure.data)
room_figure.add_trace(sound_isosurface(volume_frames[0], show_scale=True))

animation_frames = [
    go.Frame(
        name=str(index),
        data=[sound_isosurface(frame, show_scale=False)],
        traces=[sound_trace_index],
        layout=go.Layout(title=f"3D-lydudbredelse, t = {frame_times[index]:.4f} s"),
    )
    for index, frame in enumerate(volume_frames)
]

room_figure.frames = animation_frames
room_figure.update_layout(
    title=f"3D-lydudbredelse, t = {frame_times[0]:.4f} s",
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.03,
            y=0.02,
            buttons=[
                dict(
                    label="Afspil",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=80, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Tid: "),
            pad=dict(t=40),
            steps=[
                dict(
                    label=f"{time_value:.3f} s",
                    method="animate",
                    args=[
                        [str(index)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                )
                for index, time_value in enumerate(frame_times)
            ],
        )
    ],
)

room_figure.show()

