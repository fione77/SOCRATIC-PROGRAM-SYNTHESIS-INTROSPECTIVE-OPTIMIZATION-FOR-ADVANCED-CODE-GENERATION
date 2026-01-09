"""
Socratic Method Pass@k Score Calculator
Generates multiple Socratic solutions and calculates pass@1, pass@5 scores
"""
import os
import sys
import time
import json
import ast
import re
import traceback
import math
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import from your files
try:
    from leetcode import LEETCODE_HARD_PROBLEMS
    print(f"✅ Loaded {len(LEETCODE_HARD_PROBLEMS)} LeetCode hard problems")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

try:
    from socratic_gen import SocraticCodeGenerator
    print("✅ Loaded SocraticCodeGenerator")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)


class SocraticPassAtKEvaluator:
    """Evaluator for calculating pass@k scores for Socratic method"""
    
    def __init__(self):
        self.generator = SocraticCodeGenerator()
    
    def generate_multiple_solutions(self, problem: Dict, n: int = 5) -> List[Dict]:
        """
        Generate multiple Socratic solutions for a problem
        Args:
            problem: Problem definition
            n: Number of solutions to generate
        Returns:
            List of solution dictionaries with code and metadata
        """
        solutions = []
        print(f"  Generating {n} Socratic solutions...")
        
        for i in range(n):
            try:
                start_time = time.time()
                results, _ = self.generator.generate_for_problem(problem)
                generation_time = time.time() - start_time
                
                code = results.get('code', '')
                if code:
                    solutions.append({
                        'solution_id': i,
                        'code': code,
                        'generation_time': generation_time,
                        'success': True,
                        'error': None
                    })
                    print(f"    Solution {i+1}/{n}: ✓ ({generation_time:.1f}s)")
                else:
                    solutions.append({
                        'solution_id': i,
                        'code': '',
                        'generation_time': generation_time,
                        'success': False,
                        'error': 'No code generated'
                    })
                    print(f"    Solution {i+1}/{n}: ❌ No code")
                
                # Save each solution
                self.generator.save_results(results, f"{problem['id']}_{i}")
                
                # Cooldown between generations (to avoid rate limits)
                if i < n - 1:
                    time.sleep(2)
                    
            except Exception as e:
                solutions.append({
                    'solution_id': i,
                    'code': '',
                    'generation_time': 0,
                    'success': False,
                    'error': str(e)
                })
                print(f"    Solution {i+1}/{n}: ❌ Error: {e}")
                time.sleep(1)
        
        return solutions
    
    def evaluate_solution(self, code: str, problem: Dict) -> Dict:
        """Evaluate a single solution"""
        if not code or not code.strip():
            return {
                'valid_solution': False,
                'passed_all': False,
                'passed_tests': 0,
                'total_tests': 0,
                'error': 'Empty code'
            }
        
        # Check syntax
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            return {
                'valid_solution': False,
                'passed_all': False,
                'passed_tests': 0,
                'total_tests': 0,
                'error': f'Syntax error: {e}'
            }
        
        # Run tests
        test_cases = problem.get('test_cases', [])
        if not test_cases:
            return {
                'valid_solution': syntax_valid,
                'passed_all': True,
                'passed_tests': 0,
                'total_tests': 0,
                'error': 'No test cases'
            }
        
        # Execute code
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {
                'valid_solution': False,
                'passed_all': False,
                'passed_tests': 0,
                'total_tests': len(test_cases),
                'error': f'Execution error: {e}'
            }
        
        # Find function
        func_name = self._extract_function_name(code, problem)
        if not func_name or func_name not in namespace:
            return {
                'valid_solution': False,
                'passed_all': False,
                'passed_tests': 0,
                'total_tests': len(test_cases),
                'error': f'Function {func_name} not found'
            }
        
        # Run tests
        passed = 0
        test_details = []
        
        for i, test_case in enumerate(test_cases):
            try:
                inputs = test_case.get('input', {})
                expected = test_case.get('expected')
                
                if isinstance(inputs, dict):
                    result = namespace[func_name](**inputs)
                elif isinstance(inputs, (list, tuple)):
                    result = namespace[func_name](*inputs)
                else:
                    result = namespace[func_name](inputs)
                
                # Compare results
                if self._results_equal(result, expected):
                    passed += 1
                    test_details.append({
                        'test_id': i,
                        'passed': True,
                        'expected': expected,
                        'actual': result
                    })
                else:
                    test_details.append({
                        'test_id': i,
                        'passed': False,
                        'expected': expected,
                        'actual': result
                    })
                    
            except Exception as e:
                test_details.append({
                    'test_id': i,
                    'passed': False,
                    'error': str(e)
                })
        
        return {
            'valid_solution': True,
            'passed_all': passed == len(test_cases),
            'passed_tests': passed,
            'total_tests': len(test_cases),
            'test_details': test_details,
            'error': None
        }
    
    def _extract_function_name(self, code: str, problem_data: Dict) -> Optional[str]:
        """Extract main function name from code"""
        # Try to get from problem data first
        func_sig = problem_data.get('function_signature', '')
        if func_sig:
            match = re.search(r'def\s+(\w+)', func_sig)
            if match:
                return match.group(1)
        
        # Extract from code - get the first non-main function
        matches = re.findall(r'def\s+(\w+)', code)
        for func in matches:
            if func != 'main' and not func.startswith('_'):
                return func
        
        return matches[0] if matches else None
    
    def _results_equal(self, actual, expected) -> bool:
        """Compare results with tolerance for floating point"""
        if isinstance(actual, float) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 1e-5
        
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(self._results_equal(a, e) for a, e in zip(actual, expected))
        
        return actual == expected
    
    def calculate_pass_at_k(self, passed_list: List[bool], k_values: List[int] = [1, 5]) -> Dict[int, float]:
        """
        Calculate pass@k scores
        Based on formula from Codex paper: https://arxiv.org/abs/2107.03374
        
        Args:
            passed_list: List of booleans indicating which solutions passed
            k_values: List of k values to calculate (e.g., [1, 5, 10])
        
        Returns:
            Dictionary mapping k to pass@k score
        """
        n = len(passed_list)
        c = sum(passed_list)  # Number of correct solutions
        
        results = {}
        for k in k_values:
            if n - c < k:
                results[k] = 1.0
            else:
                # Calculate: 1 - ∏_{i=0}^{c-1} (1 - k/(n-i))
                product = 1.0
                for i in range(c):
                    product *= (1.0 - k / (n - i))
                results[k] = 1.0 - product
        
        return results
    
    def evaluate_problem(self, problem: Dict, solutions_per_problem: int = 5) -> Dict:
        """
        Generate and evaluate multiple solutions for a single problem
        Returns pass@k scores and detailed results
        """
        print(f"\n{'='*80}")
        print(f"📊 PROBLEM {problem['id']}: {problem['title']}")
        print(f"{'='*80}")
        
        # Generate multiple solutions
        solutions = self.generate_multiple_solutions(problem, solutions_per_problem)
        
        # Evaluate each solution
        evaluations = []
        passed_list = []
        
        print(f"\n  Evaluating solutions...")
        for i, solution in enumerate(solutions):
            if solution['code']:
                eval_result = self.evaluate_solution(solution['code'], problem)
                evaluations.append({
                    'solution_id': i,
                    'generation_time': solution['generation_time'],
                    'code_length': len(solution['code']),
                    'success': solution['success'],
                    **eval_result
                })
                passed_list.append(eval_result['passed_all'])
                status = '✓' if eval_result['passed_all'] else '✗'
                print(f"    Solution {i+1}: {status} "
                      f"(Tests: {eval_result['passed_tests']}/{eval_result['total_tests']})")
            else:
                evaluations.append({
                    'solution_id': i,
                    'generation_time': solution['generation_time'],
                    'error': solution.get('error', 'No code'),
                    'success': False,
                    'valid_solution': False,
                    'passed_all': False,
                    'passed_tests': 0,
                    'total_tests': 0
                })
                passed_list.append(False)
                print(f"    Solution {i+1}: ❌ {solution.get('error', 'No code')}")
        
        # Calculate pass@k scores
        pass_at_k = self.calculate_pass_at_k(passed_list, [1, 5])
        
        # Calculate basic statistics
        valid_solutions = sum(1 for e in evaluations if e.get('valid_solution', False))
        total_passed = sum(passed_list)
        total_tests_passed = sum(e.get('passed_tests', 0) for e in evaluations)
        total_tests = sum(e.get('total_tests', 0) for e in evaluations if e.get('total_tests', 0) > 0)
        
        if total_tests > 0:
            avg_test_pass_rate = total_tests_passed / total_tests * 100
        else:
            avg_test_pass_rate = 0
        
        # Summary
        print(f"\n  📈 RESULTS SUMMARY:")
        print(f"    Valid solutions: {valid_solutions}/{solutions_per_problem}")
        print(f"    Solutions passing all tests: {total_passed}/{solutions_per_problem}")
        print(f"    Pass@1: {pass_at_k.get(1, 0):.3f}")
        print(f"    Pass@5: {pass_at_k.get(5, 0):.3f}")
        print(f"    Raw Pass Rate: {total_passed/solutions_per_problem:.3f}")
        print(f"    Average test pass rate: {avg_test_pass_rate:.1f}%")
        
        return {
            'problem_id': problem['id'],
            'problem_title': problem['title'],
            'solutions_per_problem': solutions_per_problem,
            'valid_solutions': valid_solutions,
            'solutions_passing_all': total_passed,
            'pass_at_k': pass_at_k,
            'raw_pass_rate': total_passed / solutions_per_problem,
            'avg_test_pass_rate': avg_test_pass_rate,
            'total_generation_time': sum(s['generation_time'] for s in solutions),
            'detailed_evaluations': evaluations,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_all_problems(self, 
                            num_problems: int = 5, 
                            solutions_per_problem: int = 5) -> Dict:
        """
        Evaluate multiple problems and calculate overall pass@k scores
        """
        print("\n" + "="*80)
        print("🧠 SOCRATIC METHOD PASS@K EVALUATION")
        print("="*80)
        print(f"Testing {num_problems} problems with {solutions_per_problem} solutions each")
        print("="*80)
        
        problems_to_test = LEETCODE_HARD_PROBLEMS[:num_problems]
        all_results = []
        all_passed_lists = []
        
        # Process each problem
        for i, problem in enumerate(problems_to_test):
            print(f"\n[{i+1}/{len(problems_to_test)}] ", end="")
            result = self.evaluate_problem(problem, solutions_per_problem)
            all_results.append(result)
            
            # Extract passed status for this problem
            problem_passed = [e.get('passed_all', False) for e in result['detailed_evaluations']]
            all_passed_lists.append(problem_passed)
            
            # Save intermediate results
            self._save_problem_results(result)
            
            # Cooldown between problems
            if i < len(problems_to_test) - 1:
                print(f"\n⏳ Cooling down (10s)...")
                time.sleep(10)
        
        # Calculate overall pass@k scores
        print(f"\n{'='*80}")
        print("📊 OVERALL RESULTS ACROSS ALL PROBLEMS")
        print("="*80)
        
        # Flatten all passed lists
        all_passed_flat = []
        for passed_list in all_passed_lists:
            all_passed_flat.extend(passed_list)
        
        overall_pass_at_k = self.calculate_pass_at_k(all_passed_flat, [1, 5])
        
        # Calculate statistics
        total_solutions = len(all_passed_flat)
        total_passed = sum(all_passed_flat)
        total_generation_time = sum(r['total_generation_time'] for r in all_results)
        
        # Calculate per-problem averages
        problem_pass_rates = []
        problem_pass_at_1 = []
        problem_pass_at_5 = []
        
        for result in all_results:
            problem_pass_rates.append(result['avg_test_pass_rate'])
            problem_pass_at_1.append(result['pass_at_k'].get(1, 0))
            problem_pass_at_5.append(result['pass_at_k'].get(5, 0))
        
        avg_pass_rate = sum(problem_pass_rates) / len(problem_pass_rates) if problem_pass_rates else 0
        avg_pass_at_1 = sum(problem_pass_at_1) / len(problem_pass_at_1) if problem_pass_at_1 else 0
        avg_pass_at_5 = sum(problem_pass_at_5) / len(problem_pass_at_5) if problem_pass_at_5 else 0
        
        print(f"Total problems tested: {len(all_results)}")
        print(f"Total solutions generated: {total_solutions}")
        print(f"Solutions passing all tests: {total_passed} ({total_passed/total_solutions*100:.1f}%)")
        print(f"Total generation time: {total_generation_time:.1f}s")
        print(f"Average time per solution: {total_generation_time/total_solutions:.1f}s")
        print(f"\nOverall Pass@1: {overall_pass_at_k.get(1, 0):.3f}")
        print(f"Overall Pass@5: {overall_pass_at_k.get(5, 0):.3f}")
        print(f"Average Pass@1 per problem: {avg_pass_at_1:.3f}")
        print(f"Average Pass@5 per problem: {avg_pass_at_5:.3f}")
        print(f"Average test pass rate per problem: {avg_pass_rate:.1f}%")
        
        # Save comprehensive results
        final_results = {
            'overall': {
                'total_problems': len(all_results),
                'solutions_per_problem': solutions_per_problem,
                'total_solutions': total_solutions,
                'total_passed': total_passed,
                'overall_pass_rate': total_passed / total_solutions if total_solutions > 0 else 0,
                'overall_pass_at_k': overall_pass_at_k,
                'total_generation_time': total_generation_time,
                'avg_time_per_solution': total_generation_time / total_solutions if total_solutions > 0 else 0,
                'avg_test_pass_rate': avg_pass_rate,
                'avg_pass_at_1': avg_pass_at_1,
                'avg_pass_at_5': avg_pass_at_5
            },
            'per_problem_results': all_results,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_problems': num_problems,
                'solutions_per_problem': solutions_per_problem
            }
        }
        
        self._save_final_results(final_results)
        
        return final_results
    
    def _save_problem_results(self, result: Dict):
        """Save individual problem results"""
        os.makedirs('socratic_pass_at_k_results', exist_ok=True)
        filename = f"socratic_pass_at_k_results/problem_{result['problem_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  ✅ Results saved to {filename}")
    
    def _save_final_results(self, results: Dict):
        """Save final aggregated results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"socratic_pass_at_k_results/final_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create a human-readable summary
        summary_file = f"socratic_pass_at_k_results/summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SOCRATIC METHOD PASS@K EVALUATION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Problems tested: {results['overall']['total_problems']}\n")
            f.write(f"Solutions per problem: {results['config']['solutions_per_problem']}\n")
            f.write(f"Total solutions: {results['overall']['total_solutions']}\n")
            f.write(f"Solutions passing all tests: {results['overall']['total_passed']}\n")
            f.write(f"Overall pass rate: {results['overall']['overall_pass_rate']*100:.1f}%\n")
            f.write(f"Overall Pass@1: {results['overall']['overall_pass_at_k'].get(1, 0):.3f}\n")
            f.write(f"Overall Pass@5: {results['overall']['overall_pass_at_k'].get(5, 0):.3f}\n")
            f.write(f"Total generation time: {results['overall']['total_generation_time']:.1f}s\n")
            f.write(f"Average time per solution: {results['overall']['avg_time_per_solution']:.1f}s\n")
            f.write(f"Average test pass rate: {results['overall']['avg_test_pass_rate']:.1f}%\n")
            f.write(f"Average Pass@1 per problem: {results['overall']['avg_pass_at_1']:.3f}\n")
            f.write(f"Average Pass@5 per problem: {results['overall']['avg_pass_at_5']:.3f}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("PER PROBLEM RESULTS:\n")
            f.write("="*80 + "\n")
            
            for problem in results['per_problem_results']:
                f.write(f"\nProblem {problem['problem_id']}: {problem['problem_title']}\n")
                f.write(f"  Valid solutions: {problem['valid_solutions']}/{problem['solutions_per_problem']}\n")
                f.write(f"  Passing all tests: {problem['solutions_passing_all']}/{problem['solutions_per_problem']}\n")
                f.write(f"  Pass@1: {problem['pass_at_k'].get(1, 0):.3f}\n")
                f.write(f"  Pass@5: {problem['pass_at_k'].get(5, 0):.3f}\n")
                f.write(f"  Raw Pass Rate: {problem['raw_pass_rate']:.3f}\n")
                f.write(f"  Test pass rate: {problem['avg_test_pass_rate']:.1f}%\n")
                f.write(f"  Generation time: {problem['total_generation_time']:.1f}s\n")
        
        print(f"\n✅ Detailed results saved to {filename}")
        print(f"✅ Summary saved to {summary_file}")


def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate pass@k scores for Socratic method')
    parser.add_argument('--problems', type=int, default=5,
                       help='Number of problems to test (default: 5)')
    parser.add_argument('--solutions', type=int, default=5,
                       help='Number of solutions per problem (default: 5)')
    parser.add_argument('--problem-id', type=int,
                       help='Test a single problem by ID')
    parser.add_argument('--output-dir', type=str, default='socratic_pass_at_k_results',
                       help='Output directory for results (default: socratic_pass_at_k_results)')
    
    args = parser.parse_args()
    
    evaluator = SocraticPassAtKEvaluator()
    
    if args.problem_id:
        # Test single problem
        problem = None
        for p in LEETCODE_HARD_PROBLEMS:
            if p['id'] == args.problem_id:
                problem = p
                break
        
        if problem:
            result = evaluator.evaluate_problem(problem, args.solutions)
            print(f"\n✅ Single problem evaluation complete!")
            print(f"   Problem: {problem['title']} (ID: {problem['id']})")
            print(f"   Solutions generated: {args.solutions}")
            print(f"   Pass@1: {result['pass_at_k'].get(1, 0):.3f}")
            print(f"   Pass@5: {result['pass_at_k'].get(5, 0):.3f}")
            print(f"   Raw Pass Rate: {result['raw_pass_rate']:.3f}")
        else:
            print(f"❌ Problem ID {args.problem_id} not found")
    else:
        # Test all problems
        results = evaluator.evaluate_all_problems(
            num_problems=args.problems,
            solutions_per_problem=args.solutions
        )
        print(f"\n✅ Evaluation complete for {args.problems} problems!")
        print(f"   Overall Pass@1: {results['overall']['overall_pass_at_k'].get(1, 0):.3f}")
        print(f"   Overall Pass@5: {results['overall']['overall_pass_at_k'].get(5, 0):.3f}")
        print(f"   Overall Pass Rate: {results['overall']['overall_pass_rate']*100:.1f}%")


if __name__ == "__main__":
    main()