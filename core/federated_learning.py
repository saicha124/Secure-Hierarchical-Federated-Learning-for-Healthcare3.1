import numpy as np
import random
from typing import List, Dict, Any, Tuple
from .crypto_primitives import DifferentialPrivacy, ShamirSecretSharing
from .byzantine_tolerance import ValidatorCommittee, ProofOfWork
from .healthcare_data import HealthcareDataSimulator

class FederatedLearningSystem:
    """Main federated learning system coordinating all components"""
    
    def __init__(self, num_healthcare_facilities: int = 8, num_fog_nodes: int = 3, committee_size: int = 5):
        self.num_healthcare_facilities = num_healthcare_facilities
        self.num_fog_nodes = num_fog_nodes
        self.committee_size = committee_size
        
        # Initialize system components
        self.healthcare_facilities = [
            HealthcareFacility(f"hc_{i}", self._get_facility_type(i)) 
            for i in range(num_healthcare_facilities)
        ]
        
        self.fog_nodes = [
            FogNode(f"fog_{i}") 
            for i in range(num_fog_nodes)
        ]
        
        self.leader_server = LeaderServer("leader_server")
        self.trusted_authority = TrustedAuthority("trusted_authority")
        
        # Initialize cryptographic components
        self.differential_privacy = DifferentialPrivacy(epsilon=0.1)
        self.secret_sharing = ShamirSecretSharing(threshold=3, num_shares=5)
        self.validator_committee = ValidatorCommittee(committee_size=committee_size)
        self.proof_of_work = ProofOfWork(difficulty=4)
        
        # Initialize data simulator
        self.data_simulator = HealthcareDataSimulator()
        
        # System state
        self.current_round = 0
        self.global_model = self._initialize_global_model()
        self.registered_participants = set()
        
    def _get_facility_type(self, index: int) -> str:
        """Assign facility types for diversity"""
        types = ["hospital", "clinic", "research_center", "emergency_center"]
        return types[index % len(types)]
    
    def _initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model parameters"""
        return {
            'weights': np.random.normal(0, 0.01, (10, 1)),
            'bias': np.zeros((1,)),
            'accuracy': 0.0,
            'loss': float('inf')
        }
    
    def register_participant(self, participant_id: str) -> bool:
        """Register a participant using proof-of-work"""
        if participant_id in self.registered_participants:
            return True
            
        try:
            challenge, nonce, hash_result = self.proof_of_work.solve_challenge(participant_id)
            if self.proof_of_work.verify_solution(participant_id, nonce, hash_result):
                self.registered_participants.add(participant_id)
                return True
        except Exception:
            pass
        
        return False
    
    def run_training_round(self, epsilon: float = 0.1, local_epochs: int = 3) -> Dict[str, Any]:
        """Execute a single federated learning round"""
        self.current_round += 1
        round_results = {
            'round': self.current_round,
            'participants': [],
            'aggregated_updates': {},
            'validation_results': {},
            'metrics': {}
        }
        
        # Phase 1: Local training at healthcare facilities
        local_updates = []
        for facility in self.healthcare_facilities:
            if facility.facility_id in self.registered_participants:
                update = facility.local_training(self.global_model, local_epochs)
                
                # Apply differential privacy
                noisy_update = self.differential_privacy.add_noise(update['gradients'])
                update['gradients'] = noisy_update
                
                local_updates.append(update)
                round_results['participants'].append(facility.facility_id)
        
        # Phase 2: Secret sharing of updates
        shared_updates = []
        for update in local_updates:
            shares = self.secret_sharing.create_shares(update['gradients'].tobytes())
            shared_updates.append({
                'facility_id': update['facility_id'],
                'shares': shares,
                'metadata': {k: v for k, v in update.items() if k != 'gradients'}
            })
        
        # Phase 3: Committee validation
        all_participants = [f.facility_id for f in self.healthcare_facilities] + \
                          [f.node_id for f in self.fog_nodes]
        committee_members = self.validator_committee.select_committee(all_participants)
        
        validated_updates = []
        for shared_update in shared_updates:
            # Simulate validation (in practice, this would involve cryptographic verification)
            is_valid = self.validator_committee.validate_shares(shared_update['shares'])
            
            if is_valid:
                validated_updates.append(shared_update)
            
            round_results['validation_results'][shared_update['facility_id']] = is_valid
        
        # Phase 4: Fog node aggregation
        fog_aggregates = []
        updates_per_fog = len(validated_updates) // self.num_fog_nodes
        
        for i, fog_node in enumerate(self.fog_nodes):
            start_idx = i * updates_per_fog
            end_idx = start_idx + updates_per_fog if i < self.num_fog_nodes - 1 else len(validated_updates)
            
            fog_updates = validated_updates[start_idx:end_idx]
            if fog_updates:
                aggregate = fog_node.aggregate_updates(fog_updates)
                fog_aggregates.append(aggregate)
        
        # Phase 5: Global aggregation at leader server
        if fog_aggregates:
            global_update = self.leader_server.global_aggregation(fog_aggregates)
            self.global_model = self._update_global_model(global_update)
            round_results['aggregated_updates'] = global_update
        
        # Calculate round metrics
        round_results['metrics'] = self._calculate_round_metrics()
        
        return round_results
    
    def _update_global_model(self, global_update: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Update global model with aggregated updates"""
        learning_rate = 0.01
        
        if 'weights' in global_update:
            self.global_model['weights'] -= learning_rate * global_update['weights']
        
        # Update metrics (simplified)
        self.global_model['accuracy'] = min(0.95, self.global_model['accuracy'] + 0.02)
        self.global_model['loss'] = max(0.1, self.global_model['loss'] * 0.95)
        
        return self.global_model
    
    def _calculate_round_metrics(self) -> Dict[str, float]:
        """Calculate metrics for the current round"""
        return {
            'global_accuracy': float(self.global_model['accuracy']),
            'global_loss': float(self.global_model['loss']),
            'num_participants': len(self.registered_participants),
            'privacy_budget_used': self.differential_privacy.epsilon * self.current_round
        }


class HealthcareFacility:
    """Represents a healthcare facility participating in federated learning"""
    
    def __init__(self, facility_id: str, facility_type: str):
        self.facility_id = facility_id
        self.facility_type = facility_type
        self.local_data = self._generate_local_data()
        self.local_model = None
        self.reputation_score = random.uniform(0.8, 1.0)
        
    def _generate_local_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic healthcare data"""
        num_samples = random.randint(100, 500)
        features = np.random.normal(0, 1, (num_samples, 10))
        labels = np.random.binomial(1, 0.3, num_samples)
        
        return {
            'features': features,
            'labels': labels,
            'num_samples': num_samples
        }
    
    def local_training(self, global_model: Dict[str, np.ndarray], epochs: int = 3) -> Dict[str, Any]:
        """Perform local training and return model updates"""
        # Simulate local training (simplified logistic regression)
        X, y = self.local_data['features'], self.local_data['labels']
        weights = global_model['weights'].copy()
        
        for _ in range(epochs):
            # Simple gradient descent step
            predictions = 1 / (1 + np.exp(-X @ weights))
            gradient = X.T @ (predictions.flatten() - y) / len(y)
            weights -= 0.01 * gradient.reshape(-1, 1)
        
        # Calculate update (difference from global model)
        weight_update = weights - global_model['weights']
        
        return {
            'facility_id': self.facility_id,
            'gradients': weight_update.flatten(),
            'num_samples': self.local_data['num_samples'],
            'local_accuracy': random.uniform(0.6, 0.9)
        }


class FogNode:
    """Represents a fog node performing partial aggregation"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.processing_capacity = random.uniform(0.7, 1.0)
        
    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate updates from assigned healthcare facilities"""
        if not updates:
            return {}
        
        # Reconstruct gradients from secret shares (simplified)
        aggregated_gradients = np.zeros_like(
            np.frombuffer(updates[0]['shares'][0], dtype=np.float64)
        )
        
        total_samples = 0
        for update in updates:
            # In practice, would reconstruct from secret shares
            # Here we simulate the reconstruction
            gradient = np.random.normal(0, 0.1, len(aggregated_gradients))
            num_samples = update['metadata']['num_samples']
            
            aggregated_gradients += gradient * num_samples
            total_samples += num_samples
        
        if total_samples > 0:
            aggregated_gradients /= total_samples
        
        return {
            'node_id': self.node_id,
            'weights': aggregated_gradients.reshape(-1, 1),
            'num_participants': len(updates),
            'total_samples': total_samples
        }


class LeaderServer:
    """Represents the leader server performing global aggregation"""
    
    def __init__(self, server_id: str):
        self.server_id = server_id
        
    def global_aggregation(self, fog_aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform global aggregation of fog node results"""
        if not fog_aggregates:
            return {}
        
        # Weighted average based on number of participants
        total_weight = sum(agg['num_participants'] for agg in fog_aggregates)
        
        if total_weight == 0:
            return {}
        
        global_weights = np.zeros_like(fog_aggregates[0]['weights'])
        
        for aggregate in fog_aggregates:
            weight = aggregate['num_participants'] / total_weight
            global_weights += weight * aggregate['weights']
        
        return {
            'weights': global_weights,
            'num_fog_nodes': len(fog_aggregates),
            'total_participants': total_weight
        }


class TrustedAuthority:
    """Represents the trusted authority managing the system"""
    
    def __init__(self, authority_id: str):
        self.authority_id = authority_id
        self.registered_entities = set()
        self.issued_keys = {}
        
    def register_entity(self, entity_id: str, entity_type: str) -> bool:
        """Register an entity in the system"""
        if entity_id not in self.registered_entities:
            self.registered_entities.add(entity_id)
            self.issued_keys[entity_id] = self._generate_keys(entity_type)
            return True
        return False
    
    def _generate_keys(self, entity_type: str) -> Dict[str, str]:
        """Generate cryptographic keys for an entity"""
        return {
            'public_key': f"pk_{entity_type}_{random.randint(1000, 9999)}",
            'private_key': f"sk_{entity_type}_{random.randint(1000, 9999)}",
            'cp_abe_attributes': self._get_attributes(entity_type)
        }
    
    def _get_attributes(self, entity_type: str) -> List[str]:
        """Get CP-ABE attributes for entity type"""
        attribute_map = {
            'hospital': ['medical_institution', 'hospital', 'patient_data'],
            'clinic': ['medical_institution', 'clinic', 'patient_data'],
            'fog': ['infrastructure', 'aggregator'],
            'leader': ['infrastructure', 'global_coordinator']
        }
        return attribute_map.get(entity_type, ['basic_access'])
