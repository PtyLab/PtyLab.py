import numpy as np
import matplotlib.pyplot as plt


def GenerateFermatSpiral2(n, c=1):
    """Generate a fermat spiral just like in matlab."""
    x = np.arange(n + 1)
    r = c * np.sqrt(x)
    theta0 = 137.508 / 180 * np.pi
    theta = x * theta0
    C = r * np.cos(theta)
    R = r * np.sin(theta)
    return np.array([R, C])


def solve_order(coords):
    """A rather naive way to fix the orientation of any grid by taking the nearest point all the time.

    There are smarter ways to do this, and it can probably go wrong but so far it does not seem to be horribly off.

    """
    initial_length = get_travel_distance(coords)
    order = np.arange(len(coords[0]))
    np.random.shuffle(order)
    coords = coords[:, order]
    Y, X = coords

    R = np.hypot(Y, X)
    all_points = list(range(len(R)))[1:]
    order = [
        0,
    ]
    for i in range(len(X) - 2):
        # get the current coordinate
        pos = coords.T[order[i]]
        # calculate the nearest one, takethat
        delta_pos = coords.T[all_points] - pos

        distances = np.hypot(delta_pos.T[0], delta_pos.T[1])
        best_idx = np.argmin(distances)
        order.append(all_points[best_idx])
        del all_points[best_idx]

    final_coords = coords.T[np.array(order)].T
    final_length = get_travel_distance(final_coords)
    print(
        f"Travel distance reduced from {initial_length} to {final_length}. Reduction: {100-100*final_length/initial_length:.2f}%"
    )
    return final_coords


def get_travel_distance(coords):
    """Get the total difference traveled."""
    diff = coords - np.roll(coords, 1, axis=-1)
    total_length = np.hypot(diff[0], diff[1]).sum()
    return total_length


def display_spiral(coordinates, probe_size_coords=1):
    """
    Make a plot of the coordinates, with an appropriately sized probe.
    :param coordinates:
    :param probe_size_coords:
    :return:
    """
    fig, ax = plt.subplots(1, 1)

    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    ax.plot(coordinates[0], coordinates[1], "-", alpha=0.2)

    patches = []
    for c in coordinates.T:
        # print(c)
        patches.append(Circle(c, radius=probe_size_coords))
    p = PatchCollection(patches)
    p.set_facecolor("r")
    p.set_alpha(0.2)
    ax.add_collection(p)
    ax.set_aspect(1)
    from matplotlib.ticker import EngFormatter

    ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="m"))
    ax.set_xlim(
        ax.set_ylim(
            coordinates.flatten().min() - probe_size_coords * 2,
            coordinates.flatten().max() + probe_size_coords * 2,
        )
    )

    plt.show()


def scale_coordinates_by_probe_size(
    coordinates: np.ndarray, probe_size: float, overlap=0.6
):
    """Scale the coordinates by a given probe size, with a desired overlap. Scales the coordinates in such a way
    that we have the required overlap.
    """
    # pick five points at random, find the nearest neighbor
    N = coordinates.shape[-1]
    indices = np.random.randint(0, N, 20)
    spacing = []
    for i in indices:
        p = coordinates[:, i]
        # find the nearest neighbor
        dpos = coordinates - p[:, None]
        mutual_distance = np.hypot(dpos[0], dpos[1])
        mutual_distance[i] = np.inf
        spacing.append(np.min(mutual_distance))
    max_spacing = np.max(spacing)
    # make sure that that matches our required overlap
    required_spacing = 2 * probe_size * (1 - overlap)

    return coordinates * required_spacing / max_spacing


def save_coordinates(coordinates, filename="spiral.npz"):
    """Save the coordinates in millimeters."""
    np.savez(filename, coordinates_mm=coordinates * 1e3)


if __name__ == "__main__":
    coordinates = GenerateFermatSpiral2(100, 1)
    probe_size = 10e-6
    coordinates = scale_coordinates_by_probe_size(coordinates, probe_size, overlap=0.8)
    coordinates = solve_order(coordinates)
    display_spiral(coordinates, probe_size_coords=probe_size)
