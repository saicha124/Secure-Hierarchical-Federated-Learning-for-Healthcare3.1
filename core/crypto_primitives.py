import numpy as np
import hashlib
import secrets
from typing import List, Tuple, Any, Dict
from cryptography.fernet import Fernet
import base64

class DifferentialPrivacy:
    """Implementation of differential privacy mechanisms"""
    
    def __init__(self, epsilon: float = 0.1, delta: float = 1e-5, sensitivity: float = 2.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_scale = self.sensitivity / self.epsilon
        
    def add_noise(self, data: np.ndarray, mechanism: str = 'gaussian') -> np.ndarray:
        """Add differential privacy noise to data"""
        if mechanism == 'gaussian':
            return self._gaussian_mechanism(data)
        elif mechanism == 'laplace':
            return self._laplace_mechanism(data)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    
    def _gaussian_mechanism(self, data: np.ndarray) -> np.ndarray:
        """Gaussian noise mechanism for (ε, δ)-differential privacy"""
        # Calculate noise scale for Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def _laplace_mechanism(self, data: np.ndarray) -> np.ndarray:
        """Laplace noise mechanism for ε-differential privacy"""
        noise = np.random.laplace(0, self.noise_scale, data.shape)
        return data + noise
    
    def privacy_loss(self, num_queries: int) -> float:
        """Calculate cumulative privacy loss"""
        return self.epsilon * num_queries
    
    def is_privacy_budget_exceeded(self, num_queries: int, max_budget: float = 1.0) -> bool:
        """Check if privacy budget is exceeded"""
        return self.privacy_loss(num_queries) > max_budget
    
    def get_privacy_guarantees(self) -> Dict[str, float]:
        """Get current privacy guarantees"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'sensitivity': self.sensitivity,
            'noise_scale': self.noise_scale
        }


class ShamirSecretSharing:
    """Implementation of Shamir's Secret Sharing scheme"""
    
    def __init__(self, threshold: int, num_shares: int, prime: int = None):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime or self._generate_prime()
        
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
    
    def _generate_prime(self) -> int:
        """Generate a large prime number for finite field arithmetic"""
        # Using a well-known large prime for demonstration
        return 2**127 - 1  # Mersenne prime
    
    def create_shares(self, secret: bytes) -> List[Tuple[int, int]]:
        """Create secret shares from input data"""
        # Convert bytes to integer
        secret_int = int.from_bytes(secret, byteorder='big')
        secret_int = secret_int % self.prime
        
        # Generate random coefficients for polynomial
        coefficients = [secret_int]
        for _ in range(self.threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))
        
        # Generate shares by evaluating polynomial at different x values
        shares = []
        for x in range(1, self.num_shares + 1):
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> bytes:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares to reconstruct")
        
        # Use first 'threshold' shares
        shares = shares[:self.threshold]
        
        # Lagrange interpolation to find secret (coefficient of x^0)
        secret_int = 0
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate modular inverse of denominator
            inv_denominator = pow(denominator, self.prime - 2, self.prime)
            lagrange_coeff = (numerator * inv_denominator) % self.prime
            
            secret_int = (secret_int + yi * lagrange_coeff) % self.prime
        
        # Convert back to bytes
        secret_bytes = secret_int.to_bytes((secret_int.bit_length() + 7) // 8, byteorder='big')
        return secret_bytes
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at given x using Horner's method"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def verify_share(self, share: Tuple[int, int], verification_data: Any = None) -> bool:
        """Verify if a share is valid (simplified verification)"""
        x, y = share
        # Basic validation checks
        return 1 <= x <= self.num_shares and 0 <= y < self.prime
    
    def get_share_info(self) -> Dict[str, int]:
        """Get information about the sharing scheme"""
        return {
            'threshold': self.threshold,
            'num_shares': self.num_shares,
            'prime': self.prime,
            'security_level': min(128, self.prime.bit_length())
        }


class CPABEAccessControl:
    """Simplified implementation of Ciphertext-Policy Attribute-Based Encryption"""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.public_parameters = self._generate_public_parameters()
        
    def _generate_master_key(self) -> bytes:
        """Generate master secret key"""
        return secrets.token_bytes(32)
    
    def _generate_public_parameters(self) -> Dict[str, Any]:
        """Generate public parameters"""
        return {
            'generator': secrets.randbelow(2**256),
            'public_key': secrets.randbelow(2**256),
            'hash_function': 'sha256'
        }
    
    def generate_private_key(self, attributes: List[str]) -> Dict[str, Any]:
        """Generate private key for given attributes"""
        # Simplified key generation based on attributes
        attribute_hash = hashlib.sha256('|'.join(sorted(attributes)).encode()).digest()
        private_key = self._derive_key(self.master_key, attribute_hash)
        
        return {
            'private_key': private_key,
            'attributes': attributes,
            'key_id': hashlib.sha256(private_key).hexdigest()[:16]
        }
    
    def encrypt(self, data: bytes, access_policy: str) -> Dict[str, Any]:
        """Encrypt data with access policy"""
        # Simplified encryption using Fernet for demonstration
        key = Fernet.generate_key()
        f = Fernet(key)
        ciphertext = f.encrypt(data)
        
        # Encrypt the key using the access policy (simplified)
        policy_hash = hashlib.sha256(access_policy.encode()).digest()
        encrypted_key = self._encrypt_key_with_policy(key, policy_hash)
        
        return {
            'ciphertext': ciphertext,
            'encrypted_key': encrypted_key,
            'access_policy': access_policy,
            'encryption_params': base64.b64encode(policy_hash).decode()
        }
    
    def decrypt(self, encrypted_data: Dict[str, Any], private_key: Dict[str, Any]) -> bytes:
        """Decrypt data using private key"""
        # Check if attributes satisfy the access policy
        if not self._satisfies_policy(private_key['attributes'], encrypted_data['access_policy']):
            raise ValueError("Attributes do not satisfy access policy")
        
        # Decrypt the symmetric key
        policy_hash = base64.b64decode(encrypted_data['encryption_params'])
        symmetric_key = self._decrypt_key_with_private_key(
            encrypted_data['encrypted_key'], 
            private_key['private_key'],
            policy_hash
        )
        
        # Decrypt the data
        f = Fernet(symmetric_key)
        plaintext = f.decrypt(encrypted_data['ciphertext'])
        
        return plaintext
    
    def _derive_key(self, master_key: bytes, attribute_hash: bytes) -> bytes:
        """Derive private key from master key and attributes"""
        return hashlib.pbkdf2_hmac('sha256', master_key, attribute_hash, 100000)[:32]
    
    def _encrypt_key_with_policy(self, key: bytes, policy_hash: bytes) -> bytes:
        """Encrypt symmetric key using access policy"""
        # Simplified: XOR with policy hash (in practice, use proper ABE encryption)
        extended_hash = (policy_hash * ((len(key) // len(policy_hash)) + 1))[:len(key)]
        return bytes(a ^ b for a, b in zip(key, extended_hash))
    
    def _decrypt_key_with_private_key(self, encrypted_key: bytes, private_key: bytes, policy_hash: bytes) -> bytes:
        """Decrypt symmetric key using private key"""
        # Simplified decryption (reverse of encryption)
        extended_hash = (policy_hash * ((len(encrypted_key) // len(policy_hash)) + 1))[:len(encrypted_key)]
        return bytes(a ^ b for a, b in zip(encrypted_key, extended_hash))
    
    def _satisfies_policy(self, attributes: List[str], policy: str) -> bool:
        """Check if attributes satisfy the access policy"""
        # Simplified policy evaluation
        # In practice, this would parse and evaluate boolean expressions
        required_attributes = set(attr.strip() for attr in policy.split(' AND '))
        user_attributes = set(attributes)
        return required_attributes.issubset(user_attributes)
    
    def get_access_control_info(self) -> Dict[str, Any]:
        """Get access control system information"""
        return {
            'encryption_scheme': 'CP-ABE',
            'key_size': 256,
            'hash_function': 'SHA-256',
            'symmetric_cipher': 'Fernet (AES-128)'
        }


class CommunicationSecurity:
    """Handles secure communication between system components"""
    
    def __init__(self):
        self.session_keys = {}
        self.message_counter = {}
        
    def establish_secure_channel(self, party_a: str, party_b: str) -> str:
        """Establish secure communication channel"""
        channel_id = f"{party_a}_{party_b}"
        session_key = Fernet.generate_key()
        self.session_keys[channel_id] = session_key
        self.message_counter[channel_id] = 0
        return channel_id
    
    def encrypt_message(self, channel_id: str, message: bytes) -> Dict[str, Any]:
        """Encrypt message for secure transmission"""
        if channel_id not in self.session_keys:
            raise ValueError(f"No secure channel found: {channel_id}")
        
        session_key = self.session_keys[channel_id]
        f = Fernet(session_key)
        
        # Add message counter for replay protection
        counter = self.message_counter[channel_id]
        message_with_counter = f"{counter}|".encode() + message
        
        encrypted_message = f.encrypt(message_with_counter)
        self.message_counter[channel_id] += 1
        
        # Generate HMAC for integrity
        hmac_key = hashlib.sha256(session_key).digest()
        hmac = hashlib.hmac.new(hmac_key, encrypted_message, hashlib.sha256).hexdigest()
        
        return {
            'encrypted_message': encrypted_message,
            'hmac': hmac,
            'counter': counter,
            'channel_id': channel_id
        }
    
    def decrypt_message(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt and verify message"""
        channel_id = encrypted_data['channel_id']
        
        if channel_id not in self.session_keys:
            raise ValueError(f"No secure channel found: {channel_id}")
        
        session_key = self.session_keys[channel_id]
        
        # Verify HMAC
        hmac_key = hashlib.sha256(session_key).digest()
        expected_hmac = hashlib.hmac.new(
            hmac_key, 
            encrypted_data['encrypted_message'], 
            hashlib.sha256
        ).hexdigest()
        
        if expected_hmac != encrypted_data['hmac']:
            raise ValueError("Message integrity verification failed")
        
        # Decrypt message
        f = Fernet(session_key)
        decrypted_message = f.decrypt(encrypted_data['encrypted_message'])
        
        # Extract and verify counter
        counter_str, message = decrypted_message.decode().split('|', 1)
        counter = int(counter_str)
        
        # Simple replay protection (in practice, use sliding window)
        if counter < encrypted_data['counter']:
            raise ValueError("Replay attack detected")
        
        return message.encode()
    
    def rotate_keys(self, channel_id: str) -> bool:
        """Rotate session keys for forward secrecy"""
        if channel_id in self.session_keys:
            new_key = Fernet.generate_key()
            self.session_keys[channel_id] = new_key
            self.message_counter[channel_id] = 0
            return True
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status of communication channels"""
        return {
            'active_channels': len(self.session_keys),
            'total_messages': sum(self.message_counter.values()),
            'encryption_algorithm': 'Fernet (AES-128)',
            'integrity_algorithm': 'HMAC-SHA256',
            'key_rotation': 'Manual/Periodic'
        }
