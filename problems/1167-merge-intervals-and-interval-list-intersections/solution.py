def interval_intervals(A, B):
    # Helper function to sort and merge one interval list
    def merge(intervals):
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda interval: interval[0])
        merged = [intervals[0][:]]

        for start, end in intervals[1:]:
            last_end = merged[-1][1]

            # Overlapping or touching intervals
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

        return merged

    return merge(A), merge(B)


def interval_intersections(A, B):
    # Clean both interval lists first
    A, B = interval_intervals(A, B)

    result = []
    i = 0
    j = 0

    while i < len(A) and j < len(B):
        # Find the shared part
        overlap_start = max(A[i][0], B[j][0])
        overlap_end = min(A[i][1], B[j][1])

        # If start <= end, an intersection exists
        if overlap_start <= overlap_end:
            result.append([overlap_start, overlap_end])

        # Move past the interval that finishes first
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result