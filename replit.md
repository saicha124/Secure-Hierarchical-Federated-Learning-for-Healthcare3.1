# Overview

This project implements a secure hierarchical federated learning system specifically designed for healthcare applications. The system addresses critical challenges in medical data collaboration by combining federated learning with advanced cryptographic techniques including differential privacy, secret sharing, and Byzantine fault tolerance. The three-tier architecture consists of healthcare facilities at the edge, fog nodes for regional aggregation, and a central leader server, all coordinated through a validator committee system to ensure secure and private model training without exposing sensitive patient data.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Application Framework
The system is built using Streamlit as the web interface framework, providing an interactive dashboard for monitoring and controlling the federated learning process. The modular architecture separates concerns into distinct components for maintainability and scalability.

## Hierarchical Federated Learning Architecture
The core system implements a three-tier hierarchical structure:
- **Healthcare Facilities**: Edge nodes that hold local patient data and perform initial model training
- **Fog Nodes**: Regional aggregators that combine models from multiple healthcare facilities
- **Leader Server**: Central coordinator that performs final global model aggregation
- **Trusted Authority**: Independent entity managing cryptographic operations and validator committee oversight

This hierarchical approach reduces communication overhead compared to flat federated learning while maintaining privacy guarantees.

## Security and Privacy Components

### Differential Privacy
Implements both Gaussian and Laplace noise mechanisms to provide mathematical privacy guarantees (ε, δ-differential privacy). The system tracks privacy budget consumption across multiple queries to prevent privacy leakage through repeated model updates.

### Shamir Secret Sharing
Protects sensitive model parameters by splitting them into cryptographic shares distributed across multiple parties. This ensures no single entity can reconstruct sensitive information without cooperation from multiple participants.

### Byzantine Fault Tolerance
Incorporates a validator committee system with reputation-based selection to handle malicious participants. The system includes:
- Proof of Work consensus for validator selection
- Reputation scoring based on validation accuracy
- Committee rotation to prevent centralization of power

## Data Simulation
The healthcare data simulator generates synthetic patient data with realistic medical correlations, supporting various facility specializations (hospitals, clinics, research centers, emergency centers). This enables testing and demonstration without requiring real patient data.

## Visualization System
Provides comprehensive visualization capabilities including:
- System architecture diagrams showing network topology
- Real-time metrics monitoring (accuracy, privacy loss, consensus status)
- Training progress visualization across federation rounds
- Security event tracking and alert systems

## State Management
Uses Streamlit's session state to maintain system consistency across user interactions, tracking federation rounds, participant metrics, and historical performance data.

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the interactive dashboard
- **NumPy**: Numerical computing for model parameters and mathematical operations
- **Pandas**: Data manipulation and analysis for healthcare datasets
- **Plotly**: Interactive visualization library for charts and network diagrams
- **NetworkX**: Graph analysis for modeling federation network topology

## Cryptographic Libraries
- **cryptography**: Provides Fernet encryption and other cryptographic primitives
- **hashlib**: Hash functions for integrity verification and proof-of-work
- **secrets**: Secure random number generation for cryptographic operations

## Machine Learning Stack
The system is designed to integrate with standard ML frameworks (TensorFlow, PyTorch) for actual model training, though the current implementation focuses on the federated coordination and security mechanisms rather than specific ML model implementations.

## Data Storage
Currently uses in-memory data structures for simulation. The architecture supports integration with secure databases for production deployment, with particular consideration for HIPAA compliance in healthcare environments.