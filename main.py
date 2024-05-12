import random
import plotly.graph_objects as go
import numpy as np


class DiceMechanic:
    def __init__(self, range: list):
        self.range = range


class Hex:
    def __init__(self, q, r, closed_borders: list = None, special_exits: dict = None):
        self.q = q
        self.r = r
        self.s = self.calculate_s()
        self.closed_borders = closed_borders
        self.special_exits = special_exits
        self.size = 1
        self.times_visited = 0

    def __str__(self):
        return f"Hex at ({self.q},{self.r},{self.s})"

    def __eq__(self, other):
        if not isinstance(other, Hex):
            return NotImplemented
        return self.q == other.q and self.r == other.r and self.s == other.s

    def calculate_s(self):
        self.s = -self.q - self.r
        return self.s

    def get_cartesian_pointy(self):
        # Convert axial coordinates to cartesian coordinates for a point up orientation hex grid
        y = self.size * (3 / 2 * self.q)
        x = self.size * (np.sqrt(3) * ((self.q / 2) + self.r))
        return x, y

    def get_cartesian_flat(self):
        # Convert axial coordinates to cartesian coordinates for a flat orientation hex grid
        x = self.size * (3 / 2 * self.q)
        y = self.size * (np.sqrt(3) / 2 * self.q + np.sqrt(3) * self.r)
        return x, y


class HexFlower:
    """A HexFlower is a collection of hexes arranged in a concentric pattern.
    Rings are numbered from 0, with the center hex being ring 0."""

    def __init__(self, rings):

        self.rings = rings
        self.hexes = [Hex(0, 0)]
        self.create_hex_flower()

    def __str__(self):
        return f"HexFlower with {len(self.hexes)} hexes"

    def create_hex_flower(self):
        n = self.rings
        for q in range(-n, n + 1):
            r1 = max(-n, -q - n)
            r2 = min(n, -q + n)
            for r in range(r1, r2 + 1):
                new_hex = Hex(q, r)
                print(f"Created hex at ({q},{r})")
                self.hexes.append(new_hex)

    def get_contained_hexes(self):
        """Return a list of all hexes in the HexFlower."""
        return self.hexes

    def get_hex(self, q, r):
        """Return the hex at the given coordinates."""
        for hex in self.hexes:
            if hex.q == q and hex.r == r:
                return hex
        print(f"No hex found at ({q},{r})")
        return None

    def get_plot_data(self):
        x_coords = []
        y_coords = []
        sizes = []
        colors = []
        labels = []

        for hex in self.hexes:
            x, y = hex.get_cartesian_flat()

            x_coords.append(x)
            y_coords.append(y)

            sizes.append(hex.times_visited * 10)  # Adjust the multiplier as needed
            colors.append(hex.times_visited)

            labels.append(
                f"Pos: ({hex.q},{hex.r},{hex.s}, Times visited: {hex.times_visited})"
            )

        return x_coords, y_coords, sizes, colors, labels


class MoveMatrix:
    def __init__(
        self,
        hex_flower: HexFlower,
        dice_roller: DiceMechanic,
        orientation: str = "flat",
        moves: dict = None,
    ):
        if moves is None and orientation == "pointy":
            moves = {
                "NE": (1, 0),
                "E": (0, 1),
                "SE": (-1, 1),
                "SW": (-1, 0),
                "W": (0, -1),
                "NW": (1, -1),
            }
        elif moves is None and orientation == "flat":
            moves = {
                "N": (0, 1),
                "NE": (1, 0),
                "SE": (1, -1),
                "S": (0, -1),
                "SW": (-1, 0),
                "NW": (-1, 1),
            }
        self.hex_flower = hex_flower
        self.moves = moves
        self.distribution: dict = {}
        self.dice_roller = dice_roller

    def set_distribution(self, distribution: dict):
        self.distribution = distribution

    def setup_roll_table(self, favourite_direction: list):
        distribution = self.calculate_distribution(favourite_direction)
        distribution = self.symmetric_peak_sort_dict(distribution)
        distribution = self.distrubution_to_diceresult_range(distribution)
        self.set_distribution(distribution)

    def calculate_distribution(self, favourite_direction: list):
        distribution: dict = {}

        assignable_results: int = (
            self.dice_roller.range[1] + 1 - self.dice_roller.range[0]
        )
        even_distributable_results: int = assignable_results // len(self.moves)
        remaining_results: int = assignable_results - even_distributable_results * len(
            self.moves
        )

        for direction in self.moves:
            distribution[direction] = even_distributable_results

        while remaining_results > 0:
            for direction in favourite_direction:
                distribution[direction] += 1
                remaining_results -= 1
                if remaining_results == 0:
                    break

        return distribution

    def distrubution_to_diceresult_range(self, reordered_items: dict):
        possible_results = list(
            range(self.dice_roller.range[0], self.dice_roller.range[1] + 1)
        )
        asigned_results = {direction: [] for direction in reordered_items}

        for direction in reordered_items:
            for _ in range(reordered_items[direction]):
                asigned_results[direction].append(possible_results.pop(0))

        return asigned_results

    def symmetric_peak_sort_dict(self, distribution: dict, ascending: bool = False):

        # Extract items and sort by value
        sorted_items = self.sort_dict_by_values(distribution)

        # Prepare to reorder items symmetrically
        reordered_items = [None] * len(sorted_items)
        start_index = 0
        end_index = len(sorted_items) - 1

        # Place items starting from the middle to the ends

        for item in sorted_items:
            if ascending:
                reordered_items[start_index] = item
                start_index += 1
            else:
                reordered_items[end_index] = item
                end_index -= 1
            ascending = not ascending
        # Create a new dictionary to maintain the order
        return dict(reordered_items)

    @staticmethod
    def sort_dict_by_values(dictionary: dict):
        return sorted(dictionary.items(), key=lambda x: x[1])


class Pointer:
    def __init__(
        self,
        hex_flower: HexFlower,
        move_matrix: MoveMatrix,
        dice_roller,
        q: int,
        r: int,
    ):
        self.hex_flower = hex_flower
        self.move_matrix = move_matrix
        self.dice_roller = dice_roller

        self.q = q
        self.r = r
        self.s = self.calculate_s()

    def calculate_s(self):
        self.s = -self.q - self.r
        return self.s

    def set_pointer_position(self, q: int, r: int):
        self.q = q
        self.r = r
        self.s = self.calculate_s()

    def move_pointer(self):
        """Move the pointer in the given direction."""

        print(f"Pointer at ({self.q},{self.r})")

        direction = self.determine_move_direction()

        if direction not in self.move_matrix.moves:
            print(f"Invalid direction: {direction}, Pointer NOT moved.")
            print("---")
            return

        move_vector_qr = self.move_matrix.moves[direction]

        if (
            self.hex_flower.get_hex(
                self.q + move_vector_qr[0], self.r + move_vector_qr[1]
            )
            is None
        ):
            temp_q, temp_r, _ = self.transform_coordinates(
                self.q, self.r, self.s, move_vector_qr
            )

            self.set_pointer_position(temp_q, temp_r)

            self.hex_flower.get_hex(self.q, self.r).times_visited += 1
            print(f"Pointer moved to opposite hex at {self.q},{self.r},{self.s}")
            print("---")
            return

        self.set_pointer_position(
            self.q + move_vector_qr[0], self.r + move_vector_qr[1]
        )
        self.hex_flower.get_hex(self.q, self.r).times_visited += 1

        print(f"Pointer moved to ({self.q},{self.r})")
        print("---")

    @staticmethod
    def transform_coordinates(q, r, s, direction):

        abs_direction = (abs(direction[0]), abs(direction[1]))

        if abs_direction == (1, 0):
            return s, r, q  # srq, applies for direction (1, 0) and (-1, 0)
        elif abs_direction == (1, 1):
            return r, q, s  # rqs, applies for direction (1, -1) and (-1, 1)
        elif abs_direction == (0, 1):
            return q, s, r  # qsr, applies for direction (0, 1) and (0, -1)
        else:
            print(f"Invalid direction: {direction}, no transformation occured")
            return q, r, s  # No transformation for unrecognized directions

    def determine_move_direction(self):
        roll = self.dice_roller.roll()
        print(self.move_matrix.distribution)
        for direction in self.move_matrix.distribution:
            if roll in self.move_matrix.distribution[direction]:
                return direction

    def get_position(self):
        hex_to_get = self.hex_flower.get_hex(self.q, self.r)
        if hex_to_get is not None:
            return hex_to_get.get_cartesian_flat()
        return None


class Dice:
    def __init__(self, faces: int, modifier: int = 0):
        self.faces = list(range(1, faces + 1))
        self.value = 1
        self.modifier = modifier

    def roll(self):
        """Roll the dice and return the value."""
        self.value = random.choice(self.faces) + self.modifier
        return self.value

    def __str__(self):
        return f"Dice with {len(self.faces)} faces and a modifier of {self.modifier}"


class DicePool:
    def __init__(self, dice_count: int, faces: int):
        self.dice = [Dice(faces) for _ in range(dice_count)]

    def __str__(self):
        return f"DicePool with {len(self.dice)} dice. Each dice has {self.dice[0].faces} faces."

    def roll_all(self):
        """Roll all dice in the pool and return the values."""
        return [dice.roll() for dice in self.dice]

    def roll_all_and_sum(self):
        """Roll all dice in the pool, return the values and sum them."""
        return sum(self.roll_all())


class MiddleOf3D20(DiceMechanic):
    def __init__(self):
        super().__init__((1, 20))
        self.dice_pool = DicePool(3, 20)

    def __str__(self):
        return f"MiddleOf3D20 with {len(self.dice_pool.dice)} dice. Each dice has {self.dice_pool.dice[0].faces} faces."

    def roll(self):
        """Roll 3 dice with 20 faces each and return the middle value."""
        rolls = self.dice_pool.roll_all()
        rolls.sort()
        print(f"Rolled {rolls}, middle value is {rolls[1]}")
        return rolls[1]


class Main:
    def __init__(self):

        self.dice_roller = MiddleOf3D20()
        self.hex_flower = HexFlower(2)
        self.move_matrix = MoveMatrix(self.hex_flower, self.dice_roller)

        self.pointer = Pointer(
            self.hex_flower,
            self.move_matrix,
            self.dice_roller,
            0,
            0,
        )

    def plot(self):
        fig = go.Figure()

        # Plot HexFlower
        x_hexes, y_hexes, sizes, colors, labels = self.hex_flower.get_plot_data()

        sizes = self.normalize_size(sizes)
        colors = self.normalize_size(colors)

        fig.add_trace(
            go.Scatter(
                x=x_hexes,
                y=y_hexes,
                mode="markers+text",
                text=labels,
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale="Hot",
                    sizemode="diameter",
                ),
                textposition="bottom center",
            )
        )

        # Plot Pointer
        pointer_pos = self.pointer.get_position()
        if pointer_pos:
            fig.add_trace(
                go.Scatter(
                    x=[pointer_pos[0]],
                    y=[pointer_pos[1]],
                    mode="markers",
                    marker=dict(size=15, color="red"),
                )
            )

        fig.update_layout(
            title="HexFlower and Pointer Movement",
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.show()

    @staticmethod
    def normalize_size(values, min_size=10, max_size=50):
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
        return normalized * (max_size - min_size) + min_size


if __name__ == "__main__":
    main = Main()
    main.move_matrix.setup_roll_table(["SW"])

    for _ in range(6):
        main.pointer.move_pointer()
        main.plot()
