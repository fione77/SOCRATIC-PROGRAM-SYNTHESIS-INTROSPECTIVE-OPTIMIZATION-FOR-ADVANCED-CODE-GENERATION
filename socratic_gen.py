"""
Socratic Code Generator - FOR LEETCODE HARD PROBLEMS - FLEXIBLE VERSION
Generates code through flexible debate and blueprint validation with hallucination check
"""
import os
import time
import requests
import re
import warnings
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()


class FlexibleDebateValidator:
    """Flexible debate with blueprint validation and hallucination check"""
    
    def __init__(self, api_caller):
        self._make_api_call = api_caller
        
        # ========== RENAMED PERSONAS ==========
        self.architect_system = """You are a CODING ARCHITECT - Design optimal solution blueprints

YOUR ROLE:
1. Analyze the problem type, constraints, and requirements
2. Design 2-3 architectural approaches with clear trade-offs
3. Recommend the most suitable architecture for given constraints
4. Outline key algorithmic patterns and data structures needed

FORMAT:
PROBLEM ARCHITECTURE:
- Problem Type: [type]
- Key Requirements: [functional requirements]
- Constraints: [time/space/memory limits]

ARCHITECTURAL APPROACHES:
1. [Approach 1 - Architecture Name]: 
   - Design: [high-level design]
   - Pros: [advantages]
   - Cons: [limitations]
   - Complexity: [time/space analysis]

2. [Approach 2 - Architecture Name]:
   - Design: [high-level design]
   - Pros: [advantages]
   - Cons: [limitations]
   - Complexity: [time/space analysis]

RECOMMENDED ARCHITECTURE:
- Selected: [approach name] 
- Reason: [why this fits requirements]
- Key Insight: [main algorithmic idea]
- Risk Factors: [what could go wrong]"""

        self.optimizer_system = """You are a CODE OPTIMIZER - Create detailed, optimized implementation plans

YOUR ROLE:
1. Based on the architectural blueprint, create concrete implementation plans
2. Design function signatures, data structures, and control flow
3. Include optimization techniques and performance considerations
4. Note edge cases and error handling strategies

FORMAT:
OPTIMIZED IMPLEMENTATION PLAN FOR: [architecture name]

IMPLEMENTATION DESIGN:
- Main function: [signature with parameters and return type]
- Key helper functions: [list with purposes]
- Data structures: [structures with memory considerations]
- Algorithm flow: [step-by-step process]

PERFORMANCE OPTIMIZATIONS:
- Time complexity: O(?) - [justification with worst-case]
- Space complexity: O(?) - [justification with memory usage]
- Optimization techniques: [e.g., memoization, pruning, etc.]

EDGE CASE HANDLING:
1. [edge case 1]: [handling strategy]
2. [edge case 2]: [handling strategy]
3. [edge case 3]: [handling strategy]

IMPLEMENTATION PSEUDOCODE:
[Detailed pseudocode showing exact algorithm steps]"""

        self.tester_system = """You are a REALITY TESTER - Validate implementation against reality

YOUR ROLE:
1. Test the implementation plan against problem requirements
2. Verify the pseudocode works with given examples
3. Identify logical gaps, errors, or misunderstandings
4. Check for hallucination (inventing requirements not in problem)

FORMAT:
VALIDATION REPORT:

REQUIREMENTS MATCH:
- Matches problem statement: ✓/✗ [explain]
- Satisfies all constraints: ✓/✗ [time/space/limits]
- Handles all examples: ✓/✗ [show test results]

HALLUCINATION CHECK:
- Invented requirements: [list any requirements not in original problem]
- Missing requirements: [list any requirements from problem not addressed]
- Assumptions made: [list any unwarranted assumptions]

LOGICAL ERRORS:
1. [error 1]: [specific issue and fix]
2. [error 2]: [specific issue and fix]

CONCRETE IMPROVEMENTS NEEDED:
[Specific changes to make implementation correct]"""
    
    def _check_hallucination(self, problem: str, architect_design: str, optimizer_plan: str, 
                           tester_feedback: str) -> Dict[str, Any]:
        """Check if the generated blueprint follows the original problem requirements"""
        
        # Extract key requirements from problem
        problem_requirements = self._extract_requirements(problem)
        
        # Extract requirements from blueprint components
        architect_reqs = self._extract_requirements_from_architect(architect_design)
        optimizer_reqs = self._extract_requirements_from_optimizer(optimizer_plan)
        
        # Combine blueprint requirements
        blueprint_requirements = list(set(architect_reqs + optimizer_reqs))
        
        # Find hallucinations (requirements added that aren't in original)
        hallucinations = []
        for req in blueprint_requirements:
            if not self._requirement_matches(req, problem_requirements):
                hallucinations.append(req)
        
        # Find missing requirements (requirements from problem not in blueprint)
        missing_requirements = []
        for req in problem_requirements:
            if not self._requirement_in_blueprint(req, blueprint_requirements):
                missing_requirements.append(req)
        
        # Determine overall hallucination status
        has_hallucinations = len(hallucinations) > 0
        has_missing_requirements = len(missing_requirements) > 0
        passed_hallucination_check = not (has_hallucinations or has_missing_requirements)
        
        # Prepare feedback for other personas
        feedback_for_personas = self._prepare_persona_feedback(
            hallucinations, missing_requirements, problem_requirements, passed_hallucination_check
        )
        
        return {
            "passed": passed_hallucination_check,
            "hallucinations": hallucinations,
            "missing_requirements": missing_requirements,
            "problem_requirements": problem_requirements,
            "blueprint_requirements": blueprint_requirements,
            "feedback_for_personas": feedback_for_personas
        }
    
    def _extract_requirements(self, problem: str) -> List[str]:
        """Extract requirements from problem statement"""
        requirements = []
        
        # Look for constraints section
        constraints_match = re.search(r'(?:Constraints|限制条件|约束条件):?\s*(.*?)(?:\n\n|\n\s*\n|$)', 
                                     problem, re.IGNORECASE | re.DOTALL)
        if constraints_match:
            constraints_text = constraints_match.group(1)
            constraints = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\n\n|$)', constraints_text)
            requirements.extend(constraints)
        
        # Look for requirements section
        req_match = re.search(r'(?:Requirements|要求|需求):?\s*(.*?)(?:\n\n|\n\s*\n|$)', 
                             problem, re.IGNORECASE | re.DOTALL)
        if req_match:
            req_text = req_match.group(1)
            reqs = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\n\n|$)', req_text)
            requirements.extend(reqs)
        
        # Extract time/space complexity mentions
        if re.search(r'time.*complexity|时间复杂度', problem, re.IGNORECASE):
            time_complexity = re.search(r'Time complexity:\s*(.*?)(?:\n|$)', problem, re.IGNORECASE)
            if time_complexity:
                requirements.append(f"Time complexity: {time_complexity.group(1)}")
        
        if re.search(r'space.*complexity|空间复杂度', problem, re.IGNORECASE):
            space_complexity = re.search(r'Space complexity:\s*(.*?)(?:\n|$)', problem, re.IGNORECASE)
            if space_complexity:
                requirements.append(f"Space complexity: {space_complexity.group(1)}")
        
        # Clean and deduplicate requirements
        requirements = [req.strip() for req in requirements if req.strip()]
        return list(set(requirements))
    
    def _extract_requirements_from_architect(self, architect_design: str) -> List[str]:
        """Extract requirements from architect's design"""
        requirements = []
        
        # Look for constraints and requirements in architect design
        if "Constraints:" in architect_design:
            constraints_section = re.search(r'Constraints:\s*(.*?)(?:\n\n|\n[A-Z]|$)', 
                                           architect_design, re.DOTALL)
            if constraints_section:
                constraints = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\n\n|$)', constraints_section.group(1))
                requirements.extend(constraints)
        
        if "Key Requirements:" in architect_design:
            reqs_section = re.search(r'Key Requirements:\s*(.*?)(?:\n\n|\n[A-Z]|$)', 
                                    architect_design, re.DOTALL)
            if reqs_section:
                reqs = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\n\n|$)', reqs_section.group(1))
                requirements.extend(reqs)
        
        # Look for complexity requirements
        complexity_matches = re.findall(r'Complexity:\s*(.*?)(?:\n|$)', architect_design)
        requirements.extend(complexity_matches)
        
        return [req.strip() for req in requirements if req.strip()]
    
    def _extract_requirements_from_optimizer(self, optimizer_plan: str) -> List[str]:
        """Extract requirements from optimizer's plan"""
        requirements = []
        
        # Look for performance requirements
        if "PERFORMANCE OPTIMIZATIONS:" in optimizer_plan:
            perf_section = re.search(r'PERFORMANCE OPTIMIZATIONS:(.*?)(?:\n[A-Z]{2,}|$)',
                                    optimizer_plan, re.DOTALL)
            if perf_section:
                time_match = re.search(r'Time complexity:\s*(.*?)(?:\n|$)', perf_section.group(1), re.IGNORECASE)
                space_match = re.search(r'Space complexity:\s*(.*?)(?:\n|$)', perf_section.group(1), re.IGNORECASE)
                if time_match:
                    requirements.append(f"Time complexity: {time_match.group(1)}")
                if space_match:
                    requirements.append(f"Space complexity: {space_match.group(1)}")
        
        # Look for edge case requirements
        if "EDGE CASE HANDLING:" in optimizer_plan:
            edge_section = re.search(r'EDGE CASE HANDLING:(.*?)(?:\n[A-Z]{2,}|$)',
                                    optimizer_plan, re.DOTALL)
            if edge_section:
                edge_cases = re.findall(r'\d+\.\s*(.*?)(?:\n\d+\.|\n\n|$)', edge_section.group(1))
                requirements.extend([f"Handle edge case: {case}" for case in edge_cases])
        
        return [req.strip() for req in requirements if req.strip()]
    
    def _requirement_matches(self, blueprint_req: str, problem_reqs: List[str]) -> bool:
        """Check if a blueprint requirement matches any problem requirement"""
        blueprint_lower = blueprint_req.lower()
        
        for problem_req in problem_reqs:
            problem_lower = problem_req.lower()
            
            # Check for time complexity match
            if "time complexity" in blueprint_lower and "time complexity" in problem_lower:
                return True
            
            # Check for space complexity match
            if "space complexity" in blueprint_lower and "space complexity" in problem_lower:
                return True
            
            # Check for similar constraints (e.g., "n <= 10^5")
            if any(char.isdigit() for char in blueprint_req):
                # Extract numbers from blueprint requirement
                blueprint_nums = re.findall(r'\d+', blueprint_req)
                problem_nums = re.findall(r'\d+', problem_req)
                
                if blueprint_nums and any(num in problem_nums for num in blueprint_nums):
                    return True
            
            # Check for keyword similarity
            common_keywords = ["constraint", "requirement", "must", "should", "need", "handle", "support"]
            blueprint_words = set(blueprint_lower.split())
            problem_words = set(problem_lower.split())
            
            if len(blueprint_words & problem_words) >= 2:
                return True
        
        return False
    
    def _requirement_in_blueprint(self, problem_req: str, blueprint_reqs: List[str]) -> bool:
        """Check if a problem requirement is in the blueprint"""
        return self._requirement_matches(problem_req, blueprint_reqs)
    
    def _prepare_persona_feedback(self, hallucinations: List[str], missing_requirements: List[str],
                                problem_requirements: List[str], passed: bool) -> Dict[str, str]:
        """Prepare feedback for each persona based on hallucination check results"""
        
        feedback = {
            "architect": "",
            "optimizer": "",
            "tester": ""
        }
        
        if passed:
            feedback["architect"] = "HALLUCINATION CHECK PASSED: Architecture follows all problem requirements."
            feedback["optimizer"] = "HALLUCINATION CHECK PASSED: Implementation plan correctly addresses all requirements."
            feedback["tester"] = "HALLUCINATION CHECK PASSED: Validation should confirm all requirements are met."
            return feedback
        
        # Architect feedback
        architect_issues = []
        for hallucination in hallucinations:
            if any(keyword in hallucination.lower() for keyword in ["architecture", "approach", "design", "complexity"]):
                architect_issues.append(f"- Remove invented requirement: '{hallucination}'")
        
        for missing in missing_requirements:
            if any(keyword in missing.lower() for keyword in ["constraint", "limit", "bound", "size"]):
                architect_issues.append(f"- Add missing requirement: '{missing}'")
        
        if architect_issues:
            feedback["architect"] = "HALLUCINATION CHECK FAILED - REQUIRED CHANGES:\n" + "\n".join(architect_issues)
        
        # Optimizer feedback
        optimizer_issues = []
        for hallucination in hallucinations:
            if any(keyword in hallucination.lower() for keyword in ["implementation", "function", "data structure", "edge case"]):
                optimizer_issues.append(f"- Remove invented requirement: '{hallucination}'")
        
        for missing in missing_requirements:
            if any(keyword in missing.lower() for keyword in ["time complexity", "space complexity", "optimization", "performance"]):
                optimizer_issues.append(f"- Add missing requirement: '{missing}'")
        
        if optimizer_issues:
            feedback["optimizer"] = "HALLUCINATION CHECK FAILED - REQUIRED CHANGES:\n" + "\n".join(optimizer_issues)
        
        # Tester feedback
        tester_issues = []
        for hallucination in hallucinations:
            tester_issues.append(f"- Verify '{hallucination}' is not actually required")
        
        for missing in missing_requirements:
            tester_issues.append(f"- Ensure '{missing}' is properly tested")
        
        if tester_issues:
            feedback["tester"] = "HALLUCINATION CHECK FAILED - TESTING FOCUS:\n" + "\n".join(tester_issues)
        
        return feedback
    
    def conduct_flexible_debate(self, problem: str) -> str:
        """Run flexible debate with blueprint validation and hallucination check"""
        print("  🔄 Starting flexible debate with hallucination check...")
        
        debate_log = []
        debate_log.append(f"PROBLEM:\n{problem}\n")
        debate_log.append("\n" + "="*80)
        debate_log.append("FLEXIBLE DEBATE WITH HALLUCINATION VALIDATION")
        debate_log.append("="*80 + "\n")
        
        # ========== PHASE 1: ARCHITECTURAL DESIGN ==========
        debate_log.append("\n--- PHASE 1: ARCHITECTURAL DESIGN ---\n")
        
        print("  🏛️  ARCHITECT designing solution architecture...")
        architect = self._make_api_call(
            f"{self.architect_system}\n\nPROBLEM:\n{problem}\n\nDesign the architectural blueprint for this problem.",
            max_tokens=1000, temperature=0.7
        )
        debate_log.append(f"🏛️  ARCHITECT (ARCHITECTURE):\n{architect}\n")
        time.sleep(2)
        
        # ========== PHASE 2: OPTIMIZED IMPLEMENTATION ==========
        debate_log.append("\n--- PHASE 2: OPTIMIZED IMPLEMENTATION ---\n")
        
        print("  ⚙️  OPTIMIZER creating implementation plan...")
        optimizer = self._make_api_call(
            f"{self.optimizer_system}\n\nARCHITECT'S DESIGN:\n{architect}\n\nPROBLEM:\n{problem}\n\nCreate an optimized implementation plan based on this architecture.",
            max_tokens=1500, temperature=0.6
        )
        debate_log.append(f"⚙️  OPTIMIZER (IMPLEMENTATION PLAN):\n{optimizer}\n")
        time.sleep(2)
        
        # ========== PHASE 3: REALITY TESTING ==========
        debate_log.append("\n--- PHASE 3: REALITY TESTING ---\n")
        
        print("  🧪 TESTER validating implementation...")
        tester = self._make_api_call(
            f"{self.tester_system}\n\nPROBLEM:\n{problem}\n\nARCHITECT'S DESIGN:\n{architect}\n\nOPTIMIZER'S PLAN:\n{optimizer}\n\nValidate this implementation plan and find issues.",
            max_tokens=1200, temperature=0.6
        )
        debate_log.append(f"🧪 TESTER (VALIDATION):\n{tester}\n")
        time.sleep(2)
        
        # ========== PHASE 4: HALLUCINATION CHECK (FUNCTION) ==========
        debate_log.append("\n--- PHASE 4: HALLUCINATION AUDIT ---\n")
        
        print("  🔍 Running hallucination check function...")
        hallucination_result = self._check_hallucination(problem, architect, optimizer, tester)
        
        hallucination_report = f"HALLUCINATION CHECK RESULTS:\n"
        hallucination_report += f"Passed: {' YES' if hallucination_result['passed'] else '❌ NO'}\n\n"
        
        if hallucination_result['hallucinations']:
            hallucination_report += "INVENTED REQUIREMENTS (HALLUCINATIONS):\n"
            for i, hallucination in enumerate(hallucination_result['hallucinations'], 1):
                hallucination_report += f"{i}. {hallucination}\n"
        
        if hallucination_result['missing_requirements']:
            hallucination_report += "\nMISSING REQUIREMENTS:\n"
            for i, missing in enumerate(hallucination_result['missing_requirements'], 1):
                hallucination_report += f"{i}. {missing}\n"
        
        debate_log.append(f"🔍 HALLUCINATION CHECK:\n{hallucination_report}\n")
        
        # Notify personas with feedback
        if not hallucination_result['passed']:
            print("   Hallucinations detected - notifying personas...")
            feedback = hallucination_result['feedback_for_personas']
            
            # Update architect with feedback
            if feedback['architect']:
                architect = self._make_api_call(
                    f"{self.architect_system}\n\nPROBLEM:\n{problem}\n\nHALLUCINATION FEEDBACK:\n{feedback['architect']}\n\n"
                    f"Previous design had hallucinations. Redesign the architecture to address these issues.",
                    max_tokens=1000, temperature=0.6
                )
                debate_log.append(f" ARCHITECT (REVISED):\n{architect}\n")
            
            # Update optimizer with feedback
            if feedback['optimizer']:
                optimizer = self._make_api_call(
                    f"{self.optimizer_system}\n\nPROBLEM:\n{problem}\n\nARCHITECT'S DESIGN:\n{architect}\n\n"
                    f"HALLUCINATION FEEDBACK:\n{feedback['optimizer']}\n\n"
                    f"Previous plan had hallucinations. Create a corrected implementation plan.",
                    max_tokens=1500, temperature=0.5
                )
                debate_log.append(f" OPTIMIZER (REVISED):\n{optimizer}\n")
            
            # Update tester with feedback
            if feedback['tester']:
                tester = self._make_api_call(
                    f"{self.tester_system}\n\nPROBLEM:\n{problem}\n\nARCHITECT'S DESIGN:\n{architect}\n\n"
                    f"OPTIMIZER'S PLAN:\n{optimizer}\n\nHALLUCINATION FEEDBACK:\n{feedback['tester']}\n\n"
                    f"Validate the corrected plan with focus on hallucination areas.",
                    max_tokens=1200, temperature=0.5
                )
                debate_log.append(f" TESTER (REVISED VALIDATION):\n{tester}\n")
        else:
            print("   No hallucinations detected")
        
        # ========== PHASE 5: FINAL SYNTHESIS ==========
        debate_log.append("\n--- PHASE 5: FINAL OPTIMIZED PLAN ---\n")
        
        print("    OPTIMIZER finalizing plan...")
        final_optimizer = self._make_api_call(
            f"{self.optimizer_system}\n\nPROBLEM:\n{problem}\n\n"
            f"ARCHITECT'S FINAL DESIGN:\n{architect}\n\n"
            f"TESTER'S VALIDATION:\n{tester}\n\n"
            f"HALLUCINATION CHECK RESULTS:\n{hallucination_report}\n\n"
            f"Create a FINAL, CORRECTED implementation plan that addresses all issues.",
            max_tokens=1500, temperature=0.4
        )
        debate_log.append(f"  OPTIMIZER (FINAL PLAN):\n{final_optimizer}\n")
        
        # ========== PHASE 6: FINAL VALIDATION ==========
        debate_log.append("\n--- PHASE 6: FINAL VALIDATION ---\n")
    
        print("   TESTER performing final validation...")
        final_tester = self._make_api_call(
            f"{self.tester_system}\n\nPROBLEM:\n{problem}\n\nFINAL IMPLEMENTATION PLAN:\n{final_optimizer}\n\n"
            f"Perform a final comprehensive validation of the corrected plan.",
            max_tokens=800, temperature=0.5
        )
        debate_log.append(f" TESTER (FINAL VALIDATION):\n{final_tester}\n")
        
        debate_log.append("\n" + "="*80)
        debate_log.append("DEBATE-VALIDATION-HALLUCINATION-CHECK COMPLETED")
        debate_log.append("="*80)
        
        return "\n".join(debate_log), final_optimizer


class SocraticCodeGenerator:
    def __init__(self):
        # CHANGE 1: Switch to OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")  # Changed from GROQ_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # Changed URL
        print(f" Socratic Generator initialized (OpenRouter - Flexible Debate with Hallucination Check)")
        
        # Test API
        test_response = self._make_api_call("Say 'OpenRouter test successful'", max_tokens=20, temperature=0.1)
        print(f" API Test: {' Working' if 'test' in test_response.lower() else f'❌ Failed: {test_response}'}")
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean LLM-generated code"""
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        return code.strip()
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float = 0.7, model: str = None) -> str:
        """Make API call to OpenRouter with retry logic"""
        if not self.api_key:
            return "Error: OPENROUTER_API_KEY not found in environment variables"
        
        # CHANGE 2: OpenRouter headers and data structure
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
            "X-Title": "Socratic Code Generator"       # Your app name
        }
        
        # CHANGE 3: Choose OpenRouter model (Llama 3.3 70B)
        if model is None:
            # Using the free tier model first
            model = "meta-llama/llama-3.3-70b-instruct"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    timeout=120  # Longer timeout for larger models
                )
                
                # CHANGE 4: Handle OpenRouter-specific rate limits
                if response.status_code == 429:
                    error_msg = response.json().get("error", {}).get("message", "")
                    if "free tier" in error_msg.lower() and attempt == 0:
                        print("    Free tier limit, switching to paid model...")
                        # Switch to paid model
                        data["model"] = model.replace(":free", "")
                        continue
                    
                    wait_time = 5 * (attempt + 1)
                    print(f"     Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter Error {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f": {error_data['error'].get('message', 'Unknown')}"
                    except:
                        pass
                    print(f"     {error_msg}")
                    
                    # Fallback to Groq if OpenRouter fails
                    if attempt == 2:
                        return self._fallback_to_groq(prompt, max_tokens, temperature)
                    continue
                
                response_data = response.json()
                if "choices" not in response_data or not response_data["choices"]:
                    return "Error: No choices in response"
                
                content = response_data["choices"][0]["message"]["content"].strip()
                return content if content else "Error: Empty response content"
                
            except Exception as e:
                print(f"    Exception: {str(e)[:100]}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                return f"Error: {str(e)[:100]}"
        
        return "Error: Max retries exceeded"
    
    def _fallback_to_groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Fallback to Groq if OpenRouter fails"""
        print("     Falling back to Groq...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: No fallback API available"
        
        groq_headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
        groq_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=groq_headers,
                json=groq_data,
                timeout=90
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            return f"Groq fallback failed: {str(e)[:100]}"
        
        return "Error: All API providers failed"
    
    def _generate_code_from_optimizer(self, problem: str, optimizer_plan: str) -> str:
        """Generate code from validated optimizer plan"""
        prompt = f"""Convert this OPTIMIZED IMPLEMENTATION PLAN into complete, executable Python code:

PROBLEM:
{problem}

VALIDATED OPTIMIZER PLAN:
{optimizer_plan}

Write complete, production-ready Python code that implements the optimizer plan exactly.
Follow these guidelines:

1. IMPLEMENTATION FIDELITY:
   - Follow the optimizer plan structure EXACTLY
   - Implement ALL helper functions mentioned
   - Use EXACT data structures specified
   - Maintain EXACT time/space complexity

2. CODE QUALITY:
   - Add comprehensive docstrings explaining algorithm
   - Include type hints for all functions
   - Add comments for complex logic
   - Handle all edge cases mentioned in plan

3. ERROR HANDLING:
   - Add input validation
   - Raise appropriate exceptions for invalid inputs
   - Include meaningful error messages

4. TEST READINESS:
   - Make code easy to test
   - Ensure functions are pure where possible
   - Avoid global state

Return ONLY the Python code, no explanations.

CODE:"""
        
        # Use a model good at code generation
        code = self._make_api_call(prompt, max_tokens=2500, temperature=0.3, 
                                  model="meta-llama/llama-3.3-70b-instruct")
        return self._clean_generated_code(code)
    
    def generate_for_problem(self, problem_data: dict) -> Tuple[dict, float]:
        """Full Socratic generation with flexible debate and hallucination check"""
        problem_id = problem_data.get("id", "Unknown")
        title = problem_data.get("title", "")
        description = problem_data.get("description", "")
        
        print(f"\n{'='*80}")
        print(f"SOCRATIC GENERATION WITH HALLUCINATION CHECK - Problem {problem_id}: {title}")
        print(f"{'='*80}")
        
        total_start = time.time()
        results = {}
        
        full_problem = f"{title}\n\n{description}"
        
        # Step 1: Flexible debate with hallucination check
        print("\n Conducting flexible debate with hallucination validation...")
        debate_manager = FlexibleDebateValidator(self._make_api_call)
        debate_log, final_optimizer_plan = debate_manager.conduct_flexible_debate(full_problem)
        results['debate_log'] = debate_log
        results['optimizer_plan'] = final_optimizer_plan
        
        print(f"  ✓ Debate completed, optimized plan ready ({len(final_optimizer_plan)} chars)")
        time.sleep(2)
        
        # Step 2: Generate code from validated optimizer plan
        print("\n  Generating code from validated optimizer plan...")
        code = self._generate_code_from_optimizer(full_problem, final_optimizer_plan)
        
        # Enhanced syntax validation with multiple retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import ast
                ast.parse(code)
                print(f"   Syntax valid on attempt {attempt + 1}")
                break  # Syntax is valid
            except SyntaxError as e:
                print(f"    Syntax error, retry {attempt + 1}/{max_retries}...")
                print(f"  Error: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    # Retry with more specific prompt
                    retry_prompt = f"""Fix syntax errors in this Python code:

PROBLEM:
{full_problem}

OPTIMIZER PLAN:
{final_optimizer_plan}

CODE WITH ERRORS:
{code}

SYNTAX ERROR: {str(e)}

Provide ONLY the corrected Python code with proper syntax, no explanations:"""
                    
                    code = self._make_api_call(retry_prompt, max_tokens=2500, temperature=0.2)
                    code = self._clean_generated_code(code)
                else:
                    print("   Max retries reached for syntax errors")
        
        results['code'] = code
        results['total_time'] = time.time() - total_start
        
        print(f"\n Generation completed in {results['total_time']:.1f}s")
        print(f"  Code length: {len(code)} characters")
        print(f"  Debate log length: {len(debate_log)} characters")
        print(f"  Optimizer plan length: {len(final_optimizer_plan)} characters")
        
        return results, results['total_time']
    
    def save_results(self, results: dict, problem_id: int):
        """Save generated code and debate"""
        # Save code
        code_file = f"socratic_{problem_id}.py"
        with open(code_file, "w", encoding='utf-8') as f:
            f.write(f"# SOCRATIC GENERATION WITH HALLUCINATION CHECK - Problem {problem_id}\n")
            f.write("# Generated from validated optimizer plan (OpenRouter)\n\n")
            f.write(results['code'])
        print(f"  ✓ Saved code to {code_file}")
        
        # Save optimizer plan
        optimizer_file = f"optimizer_plan_{problem_id}.txt"
        with open(optimizer_file, "w", encoding='utf-8') as f:
            f.write(f"VALIDATED OPTIMIZER PLAN - Problem {problem_id}\n")
            f.write("="*80 + "\n\n")
            f.write(results.get('optimizer_plan', 'No optimizer plan generated'))
        print(f"  ✓ Saved optimizer plan to {optimizer_file}")
        
        # Save debate log
        debate_file = f"debate_log_{problem_id}.txt"
        with open(debate_file, "w", encoding='utf-8') as f:
            f.write(f"DEBATE LOG WITH HALLUCINATION CHECK - Problem {problem_id}\n")
            f.write("="*80 + "\n\n")
            f.write(results.get('debate_log', 'No debate log generated'))
        print(f"  ✓ Saved debate log to {debate_file}")


def test_openrouter_connection():
    """Test OpenRouter connection before running main"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file")
        print("Please add: OPENROUTER_API_KEY=your-key-here")
        return False
    
    print("✅ OPENROUTER_API_KEY found")
    
    # Quick test
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [{"role": "user", "content": "Say 'test successful'"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ OpenRouter connection successful!")
            return True
        else:
            print(f" OpenRouter error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f" Connection failed: {e}")
        return False

def generate_for_humaneval(problem_prompt: str, problem_id: int) -> str:
    """Simple wrapper for EvalPlus compatibility"""
    generator = SocraticCodeGenerator()
    return generator.generate_for_humaneval(problem_prompt)


if __name__ == "__main__":
    # Test connection first
    if not test_openrouter_connection():
        print("\n  OpenRouter connection failed. Please check:")
        print("1. Your .env file has OPENROUTER_API_KEY")
        print("2. You have credits on OpenRouter")
        print("3. Your internet connection is working")
        exit(1)
    
    generator = SocraticCodeGenerator()
    
    # Example problem
    test_problem = {
        "id": 42,
        "title": "Trapping Rain Water",
        "description": """Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5

Requirements:
- Time complexity: O(n)
- Space complexity: O(1) ideally, O(n) acceptable"""
    }
    
    results, timing = generator.generate_for_problem(test_problem)
    
    print("\n" + "="*80)
    print("CODE PREVIEW (First 500 chars):")
    print("="*80)
    print(results['code'][:500] + "..." if len(results['code']) > 500 else results['code'])
    
    print(f"\nTotal time: {timing:.2f}s")
    
    generator.save_results(results, test_problem["id"])