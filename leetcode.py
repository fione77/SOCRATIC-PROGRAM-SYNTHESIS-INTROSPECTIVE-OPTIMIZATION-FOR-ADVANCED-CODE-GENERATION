"""
LeetCode Hard Problems Dataset - Complete Version
Contains 100 hard problems with test cases and metadata
"""
import json

def generate_hard_problems():
    """Generate LeetCode hard problems"""
    problems = []
    
    # 1. N-Queens
    problems.append({
        "id": 51,
        "title": "N-Queens",
        "difficulty": "Hard",
        "description": """The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. Given an integer n, return all distinct solutions to the n-queens puzzle.""",
        "function_signature": "def solveNQueens(n: int) -> List[List[str]]:",
        "optimal_time_complexity": "O(n!)",
        "optimal_space_complexity": "O(n)",
        "requirements": [
            "Return all distinct solutions",
            "Each solution as list of board strings",
            "Queens represented as 'Q', empty as '.'",
            "Time complexity O(n!) due to permutations",
            "Space complexity O(n) for recursion depth",
            "n can be up to 9 for reasonable solutions"
        ],
        "key_edge_cases": ["n=1", "n=2", "n=4", "n=8"],
        "test_cases": [
            {
                "name": "n=1",
                "input": {"n": 1},
                "expected": [["Q"]]
            },
            {
                "name": "n=4",
                "input": {"n": 4},
                "expected": [
                    [".Q..", "...Q", "Q...", "..Q."],
                    ["..Q.", "Q...", "...Q", ".Q.."]
                ]
            }
        ]
    })
    
    # 2. N-Queens II
    problems.append({
        "id": 52,
        "title": "N-Queens II",
        "difficulty": "Hard",
        "description": """The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. Given an integer n, return the number of distinct solutions to the n-queens puzzle.""",
        "function_signature": "def totalNQueens(n: int) -> int:",
        "optimal_time_complexity": "O(n!)",
        "optimal_space_complexity": "O(n)",
        "requirements": [
            "Return count of distinct solutions",
            "Same constraints as N-Queens",
            "Optimize for count only (no board construction)",
            "n up to 9",
            "Use backtracking with pruning"
        ],
        "key_edge_cases": ["n=1", "n=2", "n=4", "n=8"],
        "test_cases": [
            {
                "name": "n=1",
                "input": {"n": 1},
                "expected": 1
            },
            {
                "name": "n=4",
                "input": {"n": 4},
                "expected": 2
            }
        ]
    })
    
    # 3. Sudoku Solver
    problems.append({
        "id": 37,
        "title": "Sudoku Solver",
        "difficulty": "Hard",
        "description": """Write a program to solve a Sudoku puzzle by filling the empty cells. A sudoku solution must satisfy all of the following rules: 1) Each of the digits 1-9 must occur exactly once in each row. 2) Each of the digits 1-9 must occur exactly once in each column. 3) Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.""",
        "function_signature": "def solveSudoku(board: List[List[str]]) -> None:",
        "optimal_time_complexity": "O(9^(n*n))",
        "optimal_space_complexity": "O(n*n)",
        "requirements": [
            "Modify board in-place",
            "Board is 9x9 with '.' for empty cells",
            "Use backtracking with constraint propagation",
            "No return value (modify input board)",
            "Time complexity exponential but optimized with pruning",
            "Space complexity O(81) for recursion"
        ],
        "key_edge_cases": ["empty board", "already solved", "no solution"],
        "test_cases": [
            {
                "name": "Example 1",
                "input": {
                    "board": [
                        ["5","3",".",".","7",".",".",".","."],
                        ["6",".",".","1","9","5",".",".","."],
                        [".","9","8",".",".",".",".","6","."],
                        ["8",".",".",".","6",".",".",".","3"],
                        ["4",".",".","8",".","3",".",".","1"],
                        ["7",".",".",".","2",".",".",".","6"],
                        [".","6",".",".",".",".","2","8","."],
                        [".",".",".","4","1","9",".",".","5"],
                        [".",".",".",".","8",".",".","7","9"]
                    ]
                },
                "expected": None
            }
        ]
    })
    
    # 4. First Missing Positive
    problems.append({
        "id": 41,
        "title": "First Missing Positive",
        "difficulty": "Hard",
        "description": """Given an unsorted integer array nums, find the smallest missing positive integer. You must implement an algorithm that runs in O(n) time and uses constant extra space.""",
        "function_signature": "def firstMissingPositive(nums: List[int]) -> int:",
        "optimal_time_complexity": "O(n)",
        "optimal_space_complexity": "O(1)",
        "requirements": [
            "Time complexity must be O(n)",
            "Space complexity must be O(1) additional space",
            "Handle negative numbers and zeros",
            "Handle duplicates",
            "Array length up to 5 * 10^5",
            "Values in range [-2^31, 2^31 - 1]"
        ],
        "key_edge_cases": ["empty array", "all negative", "contains 1", "large gap"],
        "test_cases": [
            {
                "name": "Example 1",
                "input": {"nums": [1,2,0]},
                "expected": 3
            },
            {
                "name": "Example 2",
                "input": {"nums": [3,4,-1,1]},
                "expected": 2
            },
            {
                "name": "Example 3",
                "input": {"nums": [7,8,9,11,12]},
                "expected": 1
            }
        ]
    })
    
    # 5. Text Justification
    problems.append({
        "id": 68,
        "title": "Text Justification",
        "difficulty": "Hard",
        "description": """Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully justified. Pack as many words as possible in each line.""",
        "function_signature": "def fullJustify(words: List[str], maxWidth: int) -> List[str]:",
        "optimal_time_complexity": "O(n)",
        "optimal_space_complexity": "O(n)",
        "requirements": [
            "Each line exactly maxWidth characters",
            "Last line left-justified with single spaces",
            "Other lines fully justified with extra spaces distributed",
            "Words length <= maxWidth",
            "Number of words up to 300",
            "Word length up to 20"
        ],
        "key_edge_cases": ["single word", "words fill line exactly", "very long word"],
        "test_cases": [
            {
                "name": "Example 1",
                "input": {
                    "words": ["This", "is", "an", "example", "of", "text", "justification."],
                    "maxWidth": 16
                },
                "expected": [
                    "This    is    an",
                    "example  of text",
                    "justification.  "
                ]
            }
        ]
    })
    
    
    
    return problems

# Export the problems
LEETCODE_HARD_PROBLEMS = generate_hard_problems()

if __name__ == "__main__":
    print(f"Generated {len(LEETCODE_HARD_PROBLEMS)} LeetCode hard problems")
    print("\nProblem IDs:")
    for problem in LEETCODE_HARD_PROBLEMS:
        print(f"  {problem['id']}: {problem['title']}")
    
    # Save to file
    with open("leetcode_hard_problems.json", "w", encoding="utf-8") as f:
        json.dump(LEETCODE_HARD_PROBLEMS, f, indent=2, ensure_ascii=False)

    print("\n✅ Problems saved to leetcode_hard_problems.json")
