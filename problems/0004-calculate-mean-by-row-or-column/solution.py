def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must not be empty")

    if mode == "row":
        return [sum(row) / len(row) for row in matrix]
    elif mode == "column":
        num_columns = len(matrix[0])
        if any(len(row) != num_columns for row in matrix):
            raise ValueError("All rows must have the same number of columns")
        return [
            sum(matrix[row][col] for row in range(len(matrix))) / len(matrix)
            for col in range(num_columns)
        ]
    else:
        raise ValueError("Mode must be 'row' or 'column'")