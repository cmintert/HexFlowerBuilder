import random


class Hex:
    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.s = self.calculate_s()

    def __str__(self):
        return f"Hex at ({self.q},{self.r},{self.s})"

    def __eq__(self, other):
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
        """Return a list of hexes contained in the given HexFlower instance."""
        print("HexFlower with the following hexes:")
        for hex in self.hexes:
            print(hex)
        return self.hexes

    def get_hex(self, q, r):
        """Return the hex at the given coordinates."""
        for hex in self.hexes:
            if hex.q == q and hex.r == r:
                return hex
        print(f"No hex found at ({q},{r})")
        return None


class Pointer:
    def __init__(self, hex_flower: HexFlower, q: int, r: int):
        self.hex_flower = hex_flower
        self.q = q
        self.r = r

    def move_pointer(self, direction: str):
        """Move the pointer in the given direction."""
        move_matrix = MoveMatrix(self.hex_flower)

        if direction not in move_matrix.moves:
            print(f"Invalid direction: {direction}, Pointer NOT moved.")
            return
        move = move_matrix.moves[direction]
        self.q += move[0]
        self.r += move[1]
        print(f"Pointer moved to ({self.q},{self.r})")


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

    def assign_results_to_moves(
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

        distribution = self.sort_distribution(distribution)

        return distribution

    def find_center(self, min_result, max_result):
        return (min_result + max_result) // 2

    def sort_distribution(self, distribution: dict):
        return dict(
            sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        )


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
        return rolls[1]


class Main:
    def __init__(self):
        self.hex_flower = HexFlower(2)
        self.pointer = Pointer(self.hex_flower, 0, 0)
        self.move_matrix = MoveMatrix(self.hex_flower)
        print(self.hex_flower)

    def run(self):
        pass


if __name__ == "__main__":
    main = Main()
    main.run()
    main.hex_flower.get_contained_hexes()
    main.move_matrix.assign_results_to_moves(2, 12, ["NE", "W", "SE"])
