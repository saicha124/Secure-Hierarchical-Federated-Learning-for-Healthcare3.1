import random
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class ValidatorCommittee:
    """Manages validator committee selection and consensus mechanisms"""
    
    def __init__(self, committee_size: int = 5, reputation_threshold: float = 0.7):
        self.committee_size = committee_size
        self.reputation_threshold = reputation_threshold
        self.current_committee = []
        self.reputation_scores = {}
        self.validation_history = {}
        self.rotation_interval = 3600  # 1 hour in seconds
        self.last_rotation = time.time()
        
    def initialize_reputation(self, participants: List[str]) -> None:
        """Initialize reputation scores for participants"""
        for participant in participants:
            # Initialize with random reputation scores
            self.reputation_scores[participant] = random.uniform(0.7, 1.0)
            self.validation_history[participant] = {
                'correct_validations': 0,
                'total_validations': 0,
                'last_active': time.time()
            }
    
    def select_committee(self, participants: List[str]) -> List[str]:
        """Select validator committee based on reputation and randomness"""
        if not participants:
            return []
        
        # Initialize reputation if needed
        for p in participants:
            if p not in self.reputation_scores:
                self.reputation_scores[p] = random.uniform(0.8, 1.0)
        
        # Filter participants by reputation threshold
        eligible_participants = [
            p for p in participants 
            if self.reputation_scores.get(p, 0) >= self.reputation_threshold
        ]
        
        if len(eligible_participants) < self.committee_size:
            eligible_participants = participants  # Fall back to all participants
        
        # Weighted random selection based on reputation
        weights = [self.reputation_scores.get(p, 0.5) for p in eligible_participants]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(eligible_participants)] * len(eligible_participants)
        
        # Select committee members without replacement
        selected = []
        available_participants = eligible_participants.copy()
        available_weights = weights.copy()
        
        for _ in range(min(self.committee_size, len(available_participants))):
            # Weighted random selection
            choice_idx = np.random.choice(len(available_participants), p=available_weights)
            selected.append(available_participants[choice_idx])
            
            # Remove selected participant
            available_participants.pop(choice_idx)
            available_weights.pop(choice_idx)
            
            # Renormalize weights
            if available_weights:
                total = sum(available_weights)
                if total > 0:
                    available_weights = [w / total for w in available_weights]
        
        self.current_committee = selected
        self.last_rotation = time.time()
        return selected
    
    def should_rotate_committee(self) -> bool:
        """Check if committee should be rotated"""
        return time.time() - self.last_rotation > self.rotation_interval
    
    def validate_shares(self, shares: List[Any]) -> bool:
        """Validate secret shares using committee consensus"""
        if not self.current_committee:
            return False
        
        # Simulate validation by each committee member
        validation_votes = []
        
        for validator in self.current_committee:
            # Simulate Byzantine behavior (small probability of malicious validation)
            is_byzantine = random.random() < 0.05  # 5% chance of Byzantine behavior
            
            if is_byzantine:
                # Byzantine validator gives random vote
                vote = random.choice([True, False])
            else:
                # Honest validator checks share validity
                vote = self._validate_share_integrity(shares, validator)
            
            validation_votes.append(vote)
            
            # Update validation history
            if validator not in self.validation_history:
                self.validation_history[validator] = {
                    'correct_validations': 0,
                    'total_validations': 0,
                    'last_active': time.time()
                }
            
            self.validation_history[validator]['total_validations'] += 1
            self.validation_history[validator]['last_active'] = time.time()
        
        # Consensus decision (majority vote)
        approval_votes = sum(validation_votes)
        threshold = len(self.current_committee) // 2 + 1  # Majority threshold
        
        is_approved = approval_votes >= threshold
        
        # Update reputation scores based on consensus
        self._update_reputation_scores(validation_votes, is_approved)
        
        return is_approved
    
    def _validate_share_integrity(self, shares: List[Any], validator: str) -> bool:
        """Simulate share integrity validation by a validator"""
        # Simulate cryptographic verification (always return True for valid shares)
        # In practice, this would involve actual cryptographic verification
        
        # Add some randomness to simulate real-world validation uncertainty
        base_validity = 0.95  # 95% base probability of correct validation
        
        # Adjust based on validator reputation
        validator_reliability = self.reputation_scores.get(validator, 0.8)
        validation_probability = base_validity * validator_reliability
        
        return random.random() < validation_probability
    
    def _update_reputation_scores(self, votes: List[bool], consensus_result: bool) -> None:
        """Update reputation scores based on voting behavior"""
        for i, validator in enumerate(self.current_committee):
            vote = votes[i]
            
            # Update reputation based on agreement with consensus
            if vote == consensus_result:
                # Validator voted with majority - increase reputation slightly
                self.reputation_scores[validator] = min(1.0, 
                    self.reputation_scores[validator] + 0.01)
                self.validation_history[validator]['correct_validations'] += 1
            else:
                # Validator voted against majority - decrease reputation
                self.reputation_scores[validator] = max(0.0, 
                    self.reputation_scores[validator] - 0.05)
    
    def consensus_vote(self, votes: List[bool]) -> List[bool]:
        """Simulate consensus voting process"""
        if not votes:
            return []
        
        # Add some noise to simulate real validator behavior
        noisy_votes = []
        for vote in votes:
            # Small chance of vote flip due to network issues or confusion
            if random.random() < 0.02:  # 2% chance of accidental flip
                noisy_votes.append(not vote)
            else:
                noisy_votes.append(vote)
        
        return noisy_votes
    
    def get_committee_info(self) -> Dict[str, Any]:
        """Get current committee information"""
        return {
            'current_committee': self.current_committee,
            'committee_size': len(self.current_committee),
            'average_reputation': np.mean([
                self.reputation_scores.get(member, 0.5) 
                for member in self.current_committee
            ]) if self.current_committee else 0,
            'time_since_rotation': time.time() - self.last_rotation,
            'next_rotation_due': self.should_rotate_committee()
        }
    
    def get_reputation_scores(self) -> Dict[str, float]:
        """Get reputation scores for all participants"""
        return self.reputation_scores.copy()
    
    def detect_byzantine_behavior(self, participant: str) -> Dict[str, Any]:
        """Analyze participant behavior for Byzantine patterns"""
        if participant not in self.validation_history:
            return {'is_suspicious': False, 'confidence': 0.0, 'reason': 'No history'}
        
        history = self.validation_history[participant]
        reputation = self.reputation_scores.get(participant, 0.5)
        
        # Calculate suspicious behavior indicators
        total_validations = history['total_validations']
        correct_validations = history['correct_validations']
        
        if total_validations == 0:
            accuracy_rate = 0.5
        else:
            accuracy_rate = correct_validations / total_validations
        
        # Byzantine detection heuristics
        is_suspicious = False
        suspicion_reasons = []
        confidence = 0.0
        
        # Low accuracy rate
        if accuracy_rate < 0.6 and total_validations > 5:
            is_suspicious = True
            suspicion_reasons.append("Low validation accuracy")
            confidence += 0.3
        
        # Rapidly declining reputation
        if reputation < 0.4:
            is_suspicious = True
            suspicion_reasons.append("Low reputation score")
            confidence += 0.4
        
        # Inconsistent voting patterns (simplified detection)
        if total_validations > 10 and accuracy_rate < 0.3:
            is_suspicious = True
            suspicion_reasons.append("Consistently incorrect validations")
            confidence += 0.5
        
        confidence = min(1.0, confidence)
        
        return {
            'is_suspicious': is_suspicious,
            'confidence': confidence,
            'reasons': suspicion_reasons,
            'accuracy_rate': accuracy_rate,
            'total_validations': total_validations,
            'reputation': reputation
        }


class ProofOfWork:
    """Implements Proof-of-Work mechanism for Sybil resistance"""
    
    def __init__(self, difficulty: int = 4, target_time: float = 10.0):
        self.difficulty = difficulty
        self.target_time = target_time  # Target time in seconds
        self.challenge_history = {}
        
    def generate_challenge(self, participant_id: str) -> str:
        """Generate a unique challenge for participant registration"""
        timestamp = str(int(time.time()))
        nonce_seed = f"{participant_id}_{timestamp}_{random.randint(0, 1000000)}"
        challenge = hashlib.sha256(nonce_seed.encode()).hexdigest()
        
        self.challenge_history[participant_id] = {
            'challenge': challenge,
            'timestamp': timestamp,
            'difficulty': self.difficulty,
            'solved': False
        }
        
        return challenge
    
    def solve_challenge(self, participant_id: str) -> Tuple[str, int, str]:
        """Solve the proof-of-work challenge"""
        if participant_id not in self.challenge_history:
            challenge = self.generate_challenge(participant_id)
        else:
            challenge = self.challenge_history[participant_id]['challenge']
        
        target = "0" * self.difficulty
        nonce = 0
        start_time = time.time()
        
        while True:
            # Create candidate hash
            candidate = f"{challenge}{nonce}"
            hash_result = hashlib.sha256(candidate.encode()).hexdigest()
            
            # Check if hash meets difficulty requirement
            if hash_result.startswith(target):
                end_time = time.time()
                solve_time = end_time - start_time
                
                # Update challenge history
                self.challenge_history[participant_id].update({
                    'nonce': nonce,
                    'hash_result': hash_result,
                    'solve_time': solve_time,
                    'solved': True
                })
                
                return challenge, nonce, hash_result
            
            nonce += 1
            
            # Timeout mechanism to prevent infinite loops
            if time.time() - start_time > 30:  # 30 second timeout
                raise TimeoutError("Proof-of-work solving timed out")
    
    def verify_solution(self, participant_id: str, nonce: int, hash_result: str) -> bool:
        """Verify a proof-of-work solution"""
        if participant_id not in self.challenge_history:
            return False
        
        challenge_info = self.challenge_history[participant_id]
        challenge = challenge_info['challenge']
        
        # Reconstruct the hash
        candidate = f"{challenge}{nonce}"
        expected_hash = hashlib.sha256(candidate.encode()).hexdigest()
        
        # Verify hash correctness and difficulty
        target = "0" * challenge_info['difficulty']
        
        return (expected_hash == hash_result and 
                expected_hash.startswith(target))
    
    def adjust_difficulty(self, recent_solve_times: List[float]) -> None:
        """Dynamically adjust difficulty based on recent solve times"""
        if not recent_solve_times:
            return
        
        average_time = np.mean(recent_solve_times)
        
        # Adjust difficulty to maintain target solve time
        if average_time < self.target_time * 0.8:  # Too fast
            self.difficulty += 1
        elif average_time > self.target_time * 1.2:  # Too slow
            self.difficulty = max(1, self.difficulty - 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get proof-of-work statistics"""
        solved_challenges = [
            info for info in self.challenge_history.values() 
            if info.get('solved', False)
        ]
        
        if not solved_challenges:
            return {
                'total_challenges': len(self.challenge_history),
                'solved_challenges': 0,
                'average_solve_time': 0,
                'current_difficulty': self.difficulty
            }
        
        solve_times = [info['solve_time'] for info in solved_challenges]
        
        return {
            'total_challenges': len(self.challenge_history),
            'solved_challenges': len(solved_challenges),
            'average_solve_time': np.mean(solve_times),
            'min_solve_time': np.min(solve_times),
            'max_solve_time': np.max(solve_times),
            'current_difficulty': self.difficulty,
            'target_time': self.target_time
        }
    
    def is_participant_verified(self, participant_id: str) -> bool:
        """Check if participant has completed proof-of-work"""
        if participant_id not in self.challenge_history:
            return False
        
        return self.challenge_history[participant_id].get('solved', False)
    
    def get_challenge_status(self, participant_id: str) -> Dict[str, Any]:
        """Get challenge status for a specific participant"""
        if participant_id not in self.challenge_history:
            return {'status': 'no_challenge'}
        
        info = self.challenge_history[participant_id]
        
        return {
            'status': 'solved' if info.get('solved', False) else 'pending',
            'difficulty': info['difficulty'],
            'challenge': info['challenge'][:16] + '...',  # Truncated for display
            'solve_time': info.get('solve_time', None),
            'timestamp': info['timestamp']
        }


class ByzantineFaultDetector:
    """Detects and mitigates Byzantine faults in the system"""
    
    def __init__(self, detection_window: int = 10, suspicion_threshold: float = 0.6):
        self.detection_window = detection_window
        self.suspicion_threshold = suspicion_threshold
        self.participant_behavior = {}
        self.anomaly_history = []
        
    def record_participant_action(self, participant_id: str, action_type: str, 
                                expected_result: Any, actual_result: Any) -> None:
        """Record participant action for behavior analysis"""
        if participant_id not in self.participant_behavior:
            self.participant_behavior[participant_id] = []
        
        action_record = {
            'timestamp': time.time(),
            'action_type': action_type,
            'expected': expected_result,
            'actual': actual_result,
            'is_correct': expected_result == actual_result
        }
        
        self.participant_behavior[participant_id].append(action_record)
        
        # Keep only recent actions within detection window
        cutoff_time = time.time() - (self.detection_window * 60)  # Convert to seconds
        self.participant_behavior[participant_id] = [
            record for record in self.participant_behavior[participant_id]
            if record['timestamp'] > cutoff_time
        ]
    
    def detect_byzantine_participant(self, participant_id: str) -> Dict[str, Any]:
        """Detect if a participant exhibits Byzantine behavior"""
        if participant_id not in self.participant_behavior:
            return {'is_byzantine': False, 'confidence': 0.0, 'reason': 'No data'}
        
        actions = self.participant_behavior[participant_id]
        
        if not actions:
            return {'is_byzantine': False, 'confidence': 0.0, 'reason': 'No recent actions'}
        
        # Calculate error rate
        total_actions = len(actions)
        incorrect_actions = sum(1 for action in actions if not action['is_correct'])
        error_rate = incorrect_actions / total_actions
        
        # Analyze action patterns
        action_types = [action['action_type'] for action in actions]
        unique_action_types = set(action_types)
        
        # Byzantine detection heuristics
        suspicion_indicators = []
        confidence = 0.0
        
        # High error rate
        if error_rate > 0.3:
            suspicion_indicators.append(f"High error rate: {error_rate:.2%}")
            confidence += 0.4
        
        # Consistently incorrect for specific action types
        for action_type in unique_action_types:
            type_actions = [a for a in actions if a['action_type'] == action_type]
            type_error_rate = sum(1 for a in type_actions if not a['is_correct']) / len(type_actions)
            
            if type_error_rate > 0.7 and len(type_actions) >= 3:
                suspicion_indicators.append(f"Consistent errors in {action_type}: {type_error_rate:.2%}")
                confidence += 0.3
        
        # Pattern analysis: alternating correct/incorrect behavior
        if total_actions >= 6:
            pattern_score = self._analyze_behavioral_patterns(actions)
            if pattern_score > 0.8:
                suspicion_indicators.append("Suspicious behavioral patterns detected")
                confidence += 0.2
        
        confidence = min(1.0, confidence)
        is_byzantine = confidence >= self.suspicion_threshold
        
        if is_byzantine:
            self.anomaly_history.append({
                'participant_id': participant_id,
                'timestamp': time.time(),
                'confidence': confidence,
                'indicators': suspicion_indicators
            })
        
        return {
            'is_byzantine': is_byzantine,
            'confidence': confidence,
            'error_rate': error_rate,
            'total_actions': total_actions,
            'indicators': suspicion_indicators
        }
    
    def _analyze_behavioral_patterns(self, actions: List[Dict[str, Any]]) -> float:
        """Analyze behavioral patterns for suspicious activities"""
        # Look for alternating patterns that might indicate strategic Byzantine behavior
        correctness_sequence = [action['is_correct'] for action in actions[-10:]]  # Last 10 actions
        
        # Count alternations
        alternations = 0
        for i in range(1, len(correctness_sequence)):
            if correctness_sequence[i] != correctness_sequence[i-1]:
                alternations += 1
        
        # High alternation rate might indicate strategic behavior
        alternation_rate = alternations / max(1, len(correctness_sequence) - 1)
        
        # Also check for sudden behavior changes
        if len(actions) >= 10:
            first_half = actions[:len(actions)//2]
            second_half = actions[len(actions)//2:]
            
            first_error_rate = sum(1 for a in first_half if not a['is_correct']) / len(first_half)
            second_error_rate = sum(1 for a in second_half if not a['is_correct']) / len(second_half)
            
            behavior_change = abs(first_error_rate - second_error_rate)
            
            # Combine metrics
            pattern_score = (alternation_rate * 0.6) + (behavior_change * 0.4)
        else:
            pattern_score = alternation_rate
        
        return pattern_score
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health regarding Byzantine faults"""
        total_participants = len(self.participant_behavior)
        
        if total_participants == 0:
            return {
                'total_participants': 0,
                'byzantine_participants': 0,
                'system_health': 'Unknown',
                'threat_level': 'Low'
            }
        
        # Analyze all participants
        byzantine_count = 0
        high_risk_count = 0
        
        for participant_id in self.participant_behavior:
            detection_result = self.detect_byzantine_participant(participant_id)
            
            if detection_result['is_byzantine']:
                byzantine_count += 1
            elif detection_result['confidence'] > 0.3:
                high_risk_count += 1
        
        byzantine_ratio = byzantine_count / total_participants
        
        # Determine system health
        if byzantine_ratio < 0.1:
            health = 'Excellent'
            threat_level = 'Low'
        elif byzantine_ratio < 0.2:
            health = 'Good'
            threat_level = 'Medium'
        elif byzantine_ratio < 0.33:
            health = 'Warning'
            threat_level = 'High'
        else:
            health = 'Critical'
            threat_level = 'Critical'
        
        return {
            'total_participants': total_participants,
            'byzantine_participants': byzantine_count,
            'high_risk_participants': high_risk_count,
            'byzantine_ratio': byzantine_ratio,
            'system_health': health,
            'threat_level': threat_level,
            'recent_anomalies': len([
                a for a in self.anomaly_history 
                if time.time() - a['timestamp'] < 3600  # Last hour
            ])
        }
