import random


class Hex:
    def __init__(self, q, r, closed_borders: list = None):
        self.q = q
        self.r = r
        self.s = self.calculate_s()
        self.closed_borders = closed_borders

    def __str__(self):
        return f"Hex at ({self.q},{self.r},{self.s})"

    def __eq__(self, other):
        if not isinstance(other, Hex):
            return NotImplemented
        return self.q == other.q and self.r == other.r and self.s == other.s

    def calculate_s(self):
        self.s = -self.q - self.r
        return self.s


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
        for ring in range(1, self.rings + 1):
            self.create_ring(ring)

    def create_ring(self, ring):
        q, r, s = ring, -ring, 0  # Start from the NE point of each ring
        for side in range(6):
            for step in range(ring):
                self.hexes.append(Hex(q, r))
                q, r, s = -s, -q, -r  # Move clockwise around the ring
            q, r, s = r, -q - r, q  # Rotate clockwise to the next side of the hexagon

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


class MoveMatrix:
    def __init__(self, hex_flower: HexFlower, moves: dict = None):
        if moves is None:
            moves = {
                "NE": (1, -1),
                "E": (1, 0),
                "SE": (0, 1),
                "SW": (-1, 1),
                "W": (-1, 0),
                "NW": (0, -1),
            }
        self.hex_flower = hex_flower
        self.moves = moves
        self.distribution: dict = {}

    def set_distribution(self, distribution: dict):
        self.distribution = distribution

    def setup_roll_table(
        self, min_result: int, max_result: int, favourite_direction: list
    ):
        distribution: dict = {}

        assingnable_results: int = max_result + 1 - min_result
        even_distributeable_results: int = assingnable_results // len(self.moves)
        remaining_results: int = (
            assingnable_results - even_distributeable_results * len(self.moves)
        )

        for direction in self.moves:
            distribution[direction] = even_distributeable_results

        while remaining_results > 0:
            for direction in favourite_direction:
                distribution[direction] += 1
                remaining_results -= 1
                if remaining_results == 0:
                    break

        distribution = self.symmetric_peak_sort_dict(distribution)
        distribution = self.distrubution_to_diceresult_range(
            distribution, min_result, max_result
        )
        self.set_distribution(distribution)

    def distrubution_to_diceresult_range(
        self, reordered_items: dict, min_result: int, max_result: int
    ):
        possible_results = list(range(min_result, max_result + 1))
        asigned_results = {direction: [] for direction in reordered_items}

        for direction in reordered_items:
            for value in range(reordered_items[direction]):
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

    def sort_dict_by_values(self, dictionary: dict):
        return sorted(dictionary.items(), key=lambda x: x[1])


class Pointer:
    def __init__(self, hex_flower: HexFlower, move_matrix: MoveMatrix, q: int, r: int):
        self.hex_flower = hex_flower
        self.move_matrix = move_matrix
        self.q = q
        self.r = r

    def move_pointer(self, direction: str):
        """Move the pointer in the given direction."""

        if direction not in self.move_matrix.moves:
            print(f"Invalid direction: {direction}, Pointer NOT moved.")
            return
        move = self.move_matrix.moves[direction]
        self.q += move[0]
        self.r += move[1]
        print(f"Pointer moved to ({self.q},{self.r})")

    def determine_move_direction(self, move_matrix: MoveMatrix):
        roll = MiddleOf3D20().roll()
        print(move_matrix.distribution)
        for direction in move_matrix.distribution:
            if roll in move_matrix.distribution[direction]:
                print(f"Pointer moved {direction}")
                self.move_pointer(direction)
                return


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


class MiddleOf3D20:
    def __init__(self):
        self.dice_pool = DicePool(3, 20)
        self.range = (1, 20)

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
        self.hex_flower = HexFlower(2)
        self.move_matrix = MoveMatrix(self.hex_flower)
        self.pointer = Pointer(self.hex_flower, self.move_matrix, 0, 0)

    def run(self):
        pass


if __name__ == "__main__":
    main = Main()
    main.run()
    main.move_matrix.setup_roll_table(1, 20, ["NE", "E", "SE", "SW", "W", "NW"])
    for _ in range(10):
        main.pointer.determine_move_direction(main.move_matrix)
