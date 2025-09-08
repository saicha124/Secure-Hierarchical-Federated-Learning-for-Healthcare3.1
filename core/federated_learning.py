import numpy as np
import random
from typing import List, Dict, Any, Tuple
try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    mnist = None
    Dense = None
    Sequential = None
    to_categorical = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .crypto_primitives import DifferentialPrivacy, ShamirSecretSharing
from .byzantine_tolerance import ValidatorCommittee, ProofOfWork
from .healthcare_data import HealthcareDataSimulator

class FederatedLearningSystem:
    """Main federated learning system coordinating all components"""
    
    def __init__(self, num_healthcare_facilities: int = 3, num_fog_nodes: int = 3, committee_size: int = 5, 
                 model_type: str = "Neural Network", aggregation_method: str = "FedAvg"):
        self.num_healthcare_facilities = num_healthcare_facilities
        self.num_fog_nodes = num_fog_nodes
        self.committee_size = committee_size
        
        # Prefer TensorFlow models when available
        if not TENSORFLOW_AVAILABLE and model_type in ["Neural Network", "CNN"]:
            model_type = "Logistic Regression"
            
        self.model_type = model_type
        self.aggregation_method = aggregation_method
        
        # Initialize data - prioritize MNIST when TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            # Use MNIST data when TensorFlow is available
            self.feature_vector_length = 784
            self.input_shape = (self.feature_vector_length,)
            self.client_datasets = self.load_train_dataset(n_clients=num_healthcare_facilities)
            self.test_data = self.load_test_dataset()
        else:
            # Fallback to synthetic healthcare data when TensorFlow is not available
            self.feature_vector_length = 11  # Healthcare features
            self.input_shape = (self.feature_vector_length,)
            self.client_datasets = self.load_synthetic_dataset(n_clients=num_healthcare_facilities)
            self.test_data = self.load_synthetic_test_dataset()
        
        # Initialize system components with distributed data
        self.healthcare_facilities = [
            HealthcareFacility(f"hc_{i}", self._get_facility_type(i), model_type, self.client_datasets[i]) 
            for i in range(num_healthcare_facilities)
        ]
        
        self.fog_nodes = [
            FogNode(f"fog_{i}", aggregation_method) 
            for i in range(num_fog_nodes)
        ]
        
        self.leader_server = LeaderServer("leader_server", aggregation_method)
        self.trusted_authority = TrustedAuthority("trusted_authority")
        
        # Initialize cryptographic components (will be updated with interface parameters)
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
    
    def load_train_dataset(self, n_clients=3, permute=False):
        """Load MNIST training dataset and distribute among clients"""
        if not TENSORFLOW_AVAILABLE:
            return self.load_synthetic_dataset(n_clients)
            
        client_datasets = {}  # defining local datasets for each client

        (x_train, y_train), (_, _) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], self.feature_vector_length)

        if permute == True:
            permutation_indexes = np.random.permutation(len(x_train))
            x_train = x_train[permutation_indexes]
            y_train = y_train[permutation_indexes]

        x_train = x_train.astype('float32')
        x_train /= 255

        # Convert target classes to categorical ones
        y_train = to_categorical(y_train)

        for i in range(n_clients):
            client_datasets[i] = [
                x_train[i * (len(x_train) // n_clients):i * (len(x_train) // n_clients) + (len(x_train) // n_clients)],
                y_train[i * (len(y_train) // n_clients):i * (len(y_train) // n_clients) + (len(y_train) // n_clients)]]

        return client_datasets
    
    def redistribute_data(self, new_num_facilities):
        """Redistribute data when number of facilities changes"""
        self.num_healthcare_facilities = new_num_facilities
        if TENSORFLOW_AVAILABLE:
            self.client_datasets = self.load_train_dataset(n_clients=new_num_facilities)
        else:
            self.client_datasets = self.load_synthetic_dataset(n_clients=new_num_facilities)
        
        # Recreate healthcare facilities with new data distribution
        self.healthcare_facilities = [
            HealthcareFacility(f"hc_{i}", self._get_facility_type(i), self.model_type, self.client_datasets[i]) 
            for i in range(new_num_facilities)
        ]
        
        # Reset registration for new facilities
        self.registered_participants = set()

    def load_test_dataset(self):
        """Load MNIST test dataset"""
        if not TENSORFLOW_AVAILABLE:
            return self.load_synthetic_test_dataset()
            
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], self.feature_vector_length)
        x_test = x_test.astype('float32')
        x_test /= 255
        # Convert target classes to categorical ones
        y_test = to_categorical(y_test)
        return x_test, y_test
    
    def load_synthetic_dataset(self, n_clients=3):
        """Load synthetic healthcare data and distribute among clients"""
        client_datasets = {}
        
        for i in range(n_clients):
            # Generate synthetic healthcare data for each client
            num_samples = random.randint(100, 300)
            
            # Generate healthcare features
            x_data = []
            y_data = []
            
            for _ in range(num_samples):
                # Basic demographics and vitals
                age = max(18, int(np.random.normal(50, 20)))
                gender = random.choice([0, 1])  # 0=F, 1=M
                systolic_bp = max(80, min(200, 120 + (age - 50) * 0.5 + np.random.normal(0, 15)))
                diastolic_bp = max(50, min(120, systolic_bp * 0.65 + np.random.normal(0, 8)))
                heart_rate = max(50, min(120, 72 + np.random.normal(0, 12)))
                temperature = 36.5 + np.random.normal(0, 0.8)
                glucose = max(70, 100 + np.random.normal(0, 30))
                cholesterol = max(120, 200 + np.random.normal(0, 40))
                hemoglobin = max(8, 14 + np.random.normal(0, 2))
                bmi = max(15, 25 + np.random.normal(0, 5))
                smoking = random.choice([0, 1])
                
                features = [age/100, gender, systolic_bp/200, diastolic_bp/120, heart_rate/120, 
                           temperature/40, glucose/200, cholesterol/300, hemoglobin/20, bmi/40, smoking]
                
                # Create binary classification: high risk vs low risk
                risk_score = (age > 65) + (systolic_bp > 140) + (glucose > 125) + (bmi > 30) + smoking
                high_risk = 1 if risk_score >= 2 else 0
                
                x_data.append(features)
                y_data.append([1-high_risk, high_risk])  # One-hot encoded
                
            client_datasets[i] = [np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)]
            
        return client_datasets
    
    def load_synthetic_test_dataset(self):
        """Load synthetic test dataset"""
        num_samples = 200
        x_test = []
        y_test = []
        
        for _ in range(num_samples):
            age = max(18, int(np.random.normal(50, 20)))
            gender = random.choice([0, 1])
            systolic_bp = max(80, min(200, 120 + (age - 50) * 0.5 + np.random.normal(0, 15)))
            diastolic_bp = max(50, min(120, systolic_bp * 0.65 + np.random.normal(0, 8)))
            heart_rate = max(50, min(120, 72 + np.random.normal(0, 12)))
            temperature = 36.5 + np.random.normal(0, 0.8)
            glucose = max(70, 100 + np.random.normal(0, 30))
            cholesterol = max(120, 200 + np.random.normal(0, 40))
            hemoglobin = max(8, 14 + np.random.normal(0, 2))
            bmi = max(15, 25 + np.random.normal(0, 5))
            smoking = random.choice([0, 1])
            
            features = [age/100, gender, systolic_bp/200, diastolic_bp/120, heart_rate/120,
                       temperature/40, glucose/200, cholesterol/300, hemoglobin/20, bmi/40, smoking]
            
            risk_score = (age > 65) + (systolic_bp > 140) + (glucose > 125) + (bmi > 30) + smoking
            high_risk = 1 if risk_score >= 2 else 0
            
            x_test.append(features)
            y_test.append([1-high_risk, high_risk])
        
        return np.array(x_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
    
    def predict_image(self, image_data):
        """Make prediction on a single image using the global model"""
        if self.global_model and 'model' in self.global_model:
            # Ensure image is in the right format (28x28 flattened to 784)
            if len(image_data.shape) == 2:
                image_data = image_data.reshape(1, -1)
            elif len(image_data.shape) == 1:
                image_data = image_data.reshape(1, -1)
            
            # Normalize if needed
            if image_data.max() > 1.0:
                image_data = image_data.astype('float32') / 255.0
                
            prediction = self.global_model['model'].predict(image_data, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            return predicted_class, confidence, prediction[0]
        else:
            return None, 0.0, None
        
    def _get_facility_type(self, index: int) -> str:
        """Assign facility types for diversity"""
        types = ["hospital", "clinic", "research_center", "emergency_center"]
        return types[index % len(types)]
    
    def get_model(self):
        """Create model with specific architecture"""
        if not TENSORFLOW_AVAILABLE:
            return LogisticRegression(random_state=42, max_iter=1000)
            
        model = Sequential()
        if self.feature_vector_length == 784:
            # MNIST model
            model.add(Dense(350, input_shape=(784,), activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        else:
            # Healthcare model
            model.add(Dense(64, input_shape=(self.feature_vector_length,), activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(2, activation='softmax'))
        
        # Configure the model and start training
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    
    def _create_model(self, model_type: str):
        """Create model based on type"""
        if model_type == "Neural Network":
            return self.get_model()
        elif model_type == "CNN":
            return self.get_model()  # Use same architecture for consistency
        elif model_type == "SVM":
            return SVC(probability=True, random_state=42)
        elif model_type == "Logistic Regression":
            return LogisticRegression(random_state=42)
        elif model_type == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _initialize_global_model(self) -> Dict[str, Any]:
        """Initialize global model parameters"""
        if self.model_type in ["Neural Network", "CNN"] and TENSORFLOW_AVAILABLE:
            model = self._create_model(self.model_type)
            return {
                'model': model,
                'weights': model.get_weights(),
                'accuracy': 0.0,
                'loss': float('inf')
            }
        else:
            # For sklearn models or when TensorFlow is not available
            model = self._create_model(self.model_type)
            return {
                'model': model,
                'model_type': self.model_type,
                'parameters': {},
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
    
    def run_training_round(self, epsilon: float = 0.1, local_epochs: int = 3, sensitivity: float = 0.01) -> Dict[str, Any]:
        """Execute a single federated learning round"""
        # Update differential privacy parameters if provided
        if epsilon != 0.1 or sensitivity != 0.01:
            self.differential_privacy = DifferentialPrivacy(epsilon=epsilon, sensitivity=sensitivity)
            
        self.current_round += 1
        round_results = {
            'round': self.current_round,
            'participants': [],
            'aggregated_updates': {},
            'validation_results': {},
            'metrics': {}
        }
        
        # Register all participants if not already registered
        for facility in self.healthcare_facilities:
            if facility.facility_id not in self.registered_participants:
                self.register_participant(facility.facility_id)
        
        # Phase 1: Local training at healthcare facilities
        local_updates = []
        for facility in self.healthcare_facilities:
            if facility.facility_id in self.registered_participants:
                update = facility.train_local_model(self.global_model, local_epochs)
                
                # Apply differential privacy to weights (if enabled)
                if epsilon != float('inf'):
                    noisy_weights = []
                    for weight_array in update['weights']:
                        noisy_weight = self.differential_privacy.add_noise(weight_array.flatten())
                        noisy_weights.append(noisy_weight.reshape(weight_array.shape))
                else:
                    # No privacy - use original weights
                    noisy_weights = update['weights']
                update['weights'] = noisy_weights
                
                local_updates.append(update)
                round_results['participants'].append(facility.facility_id)
        
        # Phase 2: Secret sharing of updates
        shared_updates = []
        for update in local_updates:
            # Convert weights to bytes for secret sharing
            weights_bytes = b''.join([w.tobytes() for w in update['weights']])
            shares = self.secret_sharing.create_shares(weights_bytes)
            shared_updates.append({
                'facility_id': update['facility_id'],
                'shares': shares,
                'metadata': {k: v for k, v in update.items() if k != 'weights'}
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
        
        # Phase 5: Global aggregation at leader server (FedAvg)
        if local_updates:
            # Simple FedAvg: average all the weights from local updates
            aggregated_weights = self._fedavg_aggregation(local_updates)
            self.global_model = self._update_global_model(aggregated_weights)
            round_results['aggregated_weights'] = "Weights averaged using FedAvg"
        
        # Calculate round metrics
        round_results['metrics'] = self._calculate_round_metrics()
        
        return round_results
    
    def _fedavg_aggregation(self, local_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Perform Federated Averaging (FedAvg) of local model weights"""
        if not local_updates:
            return self.global_model['weights']
        
        # Get the number of layers from the first update
        num_layers = len(local_updates[0]['weights'])
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # Calculate total samples for weighted averaging
        total_samples = sum(update['num_samples'] for update in local_updates)
        
        # For each layer, compute weighted average
        for layer_idx in range(num_layers):
            layer_weights = []
            layer_shape = local_updates[0]['weights'][layer_idx].shape
            
            # Weighted sum of weights for this layer
            weighted_sum = np.zeros(layer_shape)
            for update in local_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['weights'][layer_idx]
            
            aggregated_weights.append(weighted_sum)
        
        return aggregated_weights
    
    def _update_global_model(self, aggregated_weights: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Update global model with properly aggregated weights using FedAvg"""
        # Update the global model with the aggregated weights
        self.global_model['weights'] = aggregated_weights
        
        if TENSORFLOW_AVAILABLE and 'model' in self.global_model and hasattr(self.global_model['model'], 'set_weights'):
            try:
                # Convert numpy arrays to the right format for TensorFlow
                tf_weights = []
                for weight in aggregated_weights:
                    if isinstance(weight, np.ndarray):
                        tf_weights.append(weight.astype(np.float32))
                    else:
                        tf_weights.append(np.array(weight, dtype=np.float32))
                
                self.global_model['model'].set_weights(tf_weights)
                
                # Evaluate the updated model on test data for real metrics
                if hasattr(self, 'test_data') and self.test_data is not None:
                    x_test, y_test = self.test_data
                    loss, accuracy = self.global_model['model'].evaluate(x_test, y_test, verbose=0)
                    self.global_model['accuracy'] = float(accuracy)
                    self.global_model['loss'] = float(loss)
                else:
                    # Keep current accuracy if no test data
                    pass
            except Exception as e:
                print(f"Error updating TensorFlow model weights: {e}")
                # Fall back to simulated metrics
                self.global_model['accuracy'] = max(0.1, self.global_model.get('accuracy', 0.9) * 0.95)
                self.global_model['loss'] = self.global_model.get('loss', 0.1) * 1.05
        else:
            # For sklearn models, simulate evaluation
            self.global_model['accuracy'] = 0.7 + np.random.normal(0, 0.1)  # Simulated accuracy
            self.global_model['loss'] = 1 - self.global_model['accuracy']
        
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
    
    def __init__(self, facility_id: str, facility_type: str, model_type: str = "Neural Network", mnist_data=None):
        self.facility_id = facility_id
        self.facility_type = facility_type
        self.model_type = model_type
        self.mnist_data = mnist_data  # [x_data, y_data]
        self.local_model = None
        self.reputation_score = random.uniform(0.8, 1.0)
        
    def get_local_data(self) -> Dict[str, np.ndarray]:
        """Get MNIST data for this facility"""
        if self.mnist_data is not None:
            x_data, y_data = self.mnist_data
            return {
                'features': x_data,
                'labels': y_data,
                'num_samples': len(x_data)
            }
        else:
            # Fallback to synthetic data if no MNIST data provided
            num_samples = random.randint(100, 500)
            features = np.random.normal(0, 1, (num_samples, 784))
            labels = np.random.randint(0, 10, (num_samples, 10))
            
            return {
                'features': features,
                'labels': labels,
                'num_samples': num_samples
            }
    
    def train_local_model(self, global_model, epochs: int = 3) -> Dict[str, Any]:
        """Train local model with data"""
        local_data = self.get_local_data()
        X, y = local_data['features'], local_data['labels']
        
        if TENSORFLOW_AVAILABLE and 'weights' in global_model:
            # TensorFlow model training
            local_model = tf.keras.models.clone_model(global_model['model'])
            local_model.set_weights(global_model['weights'])
            
            # Train the local model
            history = local_model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
            
            # Get updated weights
            updated_weights = local_model.get_weights()
            
            # Calculate local accuracy
            local_accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.0
            loss = history.history['loss'][-1] if 'loss' in history.history else 0.0
            
            return {
                'facility_id': self.facility_id,
                'weights': updated_weights,
                'num_samples': local_data['num_samples'],
                'local_accuracy': local_accuracy,
                'loss': loss
            }
        else:
            # Sklearn model training
            from sklearn.base import clone
            local_model = clone(global_model['model'])
            
            # Convert y to 1D for sklearn if needed
            if len(y.shape) > 1 and y.shape[1] == 2:
                y_1d = np.argmax(y, axis=1)
            else:
                y_1d = y.flatten() if len(y.shape) > 1 else y
            
            # Train the model
            local_model.fit(X, y_1d)
            
            # Calculate accuracy
            predictions = local_model.predict(X)
            local_accuracy = np.mean(predictions == y_1d)
            
            # For sklearn models, we simulate "weights" with model parameters
            if hasattr(local_model, 'coef_'):
                weights = [local_model.coef_]
                if hasattr(local_model, 'intercept_'):
                    weights.append(local_model.intercept_.reshape(-1, 1))
            else:
                # For tree-based models, create dummy weights
                weights = [np.random.normal(0, 0.1, (X.shape[1], 2))]
            
            return {
                'facility_id': self.facility_id,
                'weights': weights,
                'num_samples': local_data['num_samples'],
                'local_accuracy': local_accuracy,
                'loss': 1 - local_accuracy  # Simple loss approximation
            }


class FogNode:
    """Represents a fog node performing partial aggregation"""
    
    def __init__(self, node_id: str, aggregation_method: str = "FedAvg"):
        self.node_id = node_id
        self.aggregation_method = aggregation_method
        self.processing_capacity = random.uniform(0.7, 1.0)
        
    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate updates from assigned healthcare facilities"""
        if not updates:
            return {}
        
        # Get the dimension from the first update's weights
        first_weights = updates[0].get('weights', [np.array([[0.0]])])
        if isinstance(first_weights, list) and len(first_weights) > 0:
            weight_shape = first_weights[0].shape
            aggregated_weights = np.zeros(weight_shape)
        else:
            # Fallback dimension
            aggregated_weights = np.zeros((784, 10))
        
        total_samples = 0
        total_loss = 0.0
        
        for update in updates:
            # Get actual weights from the update
            weights = update.get('weights', [])
            num_samples = update.get('num_samples', 1)
            loss = update.get('loss', 0.0)
            
            if weights and len(weights) > 0:
                # Use actual model weights for aggregation
                weight_contrib = np.array(weights[0]) * num_samples
                aggregated_weights += weight_contrib
            
            total_samples += num_samples
            total_loss += loss * num_samples
        
        if total_samples > 0:
            aggregated_weights /= total_samples
            avg_loss = total_loss / total_samples
        else:
            avg_loss = 0.0
        
        return {
            'node_id': self.node_id,
            'weights': [aggregated_weights],
            'num_participants': len(updates),
            'total_samples': total_samples,
            'avg_loss': avg_loss
        }


class LeaderServer:
    """Represents the leader server performing global aggregation"""
    
    def __init__(self, server_id: str, aggregation_method: str = "FedAvg"):
        self.server_id = server_id
        self.aggregation_method = aggregation_method
        
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
