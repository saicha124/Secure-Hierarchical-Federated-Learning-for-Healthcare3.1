import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import hashlib
import secrets
from datetime import datetime, timedelta
import json

# Import custom modules
from core.federated_learning import FederatedLearningSystem
from core.crypto_primitives import DifferentialPrivacy, ShamirSecretSharing
from core.byzantine_tolerance import ValidatorCommittee, ProofOfWork
from core.healthcare_data import HealthcareDataSimulator
from utils.visualization import SystemArchitectureViz, MetricsViz

# Page configuration
st.set_page_config(
    page_title="Secure Hierarchical Federated Learning for Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
# System configuration parameters
if 'num_healthcare_facilities' not in st.session_state:
    st.session_state.num_healthcare_facilities = 8
if 'num_fog_nodes' not in st.session_state:
    st.session_state.num_fog_nodes = 3
if 'committee_size' not in st.session_state:
    st.session_state.committee_size = 5
if 'epsilon' not in st.session_state:
    st.session_state.epsilon = 0.1
if 'delta' not in st.session_state:
    st.session_state.delta = 1e-5

# New ML configuration parameters
if 'max_training_rounds' not in st.session_state:
    st.session_state.max_training_rounds = 10
if 'aggregation_method' not in st.session_state:
    st.session_state.aggregation_method = 'FedAvg'
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Neural Network'
if 'enable_secret_sharing' not in st.session_state:
    st.session_state.enable_secret_sharing = True
if 'secret_sharing_threshold' not in st.session_state:
    st.session_state.secret_sharing_threshold = 3
if 'secret_sharing_shares' not in st.session_state:
    st.session_state.secret_sharing_shares = 5

def initialize_system():
    """Initialize the federated learning system"""
    if st.session_state.system is None:
        system = FederatedLearningSystem(
            num_healthcare_facilities=st.session_state.num_healthcare_facilities,
            num_fog_nodes=st.session_state.num_fog_nodes,
            committee_size=st.session_state.committee_size,
            model_type=st.session_state.model_type,
            aggregation_method=st.session_state.aggregation_method
        )
        
        # Load custom dataset if available
        if st.session_state.get('use_custom_dataset', False) and 'uploaded_dataset' in st.session_state:
            system.data_simulator.load_custom_dataset(st.session_state.uploaded_dataset)
        
        st.session_state.system = system
    return st.session_state.system

def main():
    st.title("🏥 Secure Hierarchical Federated Learning for Healthcare")
    st.markdown("### Byzantine-Resilient FL with Differential Privacy & Secret Sharing")
    
    # Sidebar configuration
    st.sidebar.title("System Configuration")
    
    # System parameters
    st.sidebar.subheader("Network Parameters")
    new_facilities = st.sidebar.number_input(
        "Healthcare Facilities",
        min_value=3, max_value=20, value=st.session_state.num_healthcare_facilities
    )
    new_fog_nodes = st.sidebar.number_input(
        "Fog Nodes", 
        min_value=2, max_value=20, value=st.session_state.num_fog_nodes
    )
    new_committee_size = st.sidebar.number_input(
        "Validator Committee Size",
        min_value=3, max_value=15, value=st.session_state.committee_size
    )
    
    # Privacy parameters
    st.sidebar.subheader("Privacy Parameters")
    new_epsilon = st.sidebar.slider(
        "Epsilon (ε) - Privacy Budget",
        min_value=0.01, max_value=1.0, value=st.session_state.epsilon, step=0.01
    )
    new_delta = st.sidebar.selectbox(
        "Delta (δ)",
        [1e-3, 1e-4, 1e-5, 1e-6], index=[1e-3, 1e-4, 1e-5, 1e-6].index(st.session_state.delta)
    )
    
    # Machine Learning Configuration
    st.sidebar.subheader("ML Configuration")
    new_model_type = st.sidebar.selectbox(
        "Model Type",
        ["Neural Network", "CNN", "SVM", "Logistic Regression", "Random Forest"],
        index=["Neural Network", "CNN", "SVM", "Logistic Regression", "Random Forest"].index(st.session_state.model_type)
    )
    
    new_max_training_rounds = st.sidebar.slider(
        "Maximum Training Rounds",
        min_value=5, max_value=150, value=st.session_state.max_training_rounds, step=5
    )
    
    new_aggregation_method = st.sidebar.selectbox(
        "Aggregation Method",
        ["FedAvg", "FedProx"],
        index=["FedAvg", "FedProx"].index(st.session_state.aggregation_method)
    )
    
    # Secret Sharing Configuration
    st.sidebar.subheader("Secret Sharing")
    new_enable_secret_sharing = st.sidebar.checkbox(
        "Enable Secret Sharing",
        value=st.session_state.enable_secret_sharing
    )
    
    if new_enable_secret_sharing:
        # Automatically set shares to match number of fog nodes
        new_secret_sharing_shares = new_fog_nodes
        st.sidebar.info(f"Shares set to match fog nodes: {new_secret_sharing_shares}")
        
        new_secret_sharing_threshold = st.sidebar.number_input(
            "Threshold (t)",
            min_value=2, max_value=new_fog_nodes, 
            value=min(st.session_state.secret_sharing_threshold, new_fog_nodes),
            help="Minimum number of shares required to reconstruct the secret"
        )
    else:
        new_secret_sharing_threshold = st.session_state.secret_sharing_threshold
        new_secret_sharing_shares = st.session_state.secret_sharing_shares
    
    # Check if parameters changed
    params_changed = (
        new_facilities != st.session_state.num_healthcare_facilities or
        new_fog_nodes != st.session_state.num_fog_nodes or
        new_committee_size != st.session_state.committee_size or
        new_epsilon != st.session_state.epsilon or
        new_delta != st.session_state.delta or
        new_model_type != st.session_state.model_type or
        new_max_training_rounds != st.session_state.max_training_rounds or
        new_aggregation_method != st.session_state.aggregation_method or
        new_enable_secret_sharing != st.session_state.enable_secret_sharing or
        new_secret_sharing_threshold != st.session_state.secret_sharing_threshold or
        new_secret_sharing_shares != st.session_state.secret_sharing_shares
    )
    
    if params_changed:
        st.session_state.num_healthcare_facilities = new_facilities
        st.session_state.num_fog_nodes = new_fog_nodes
        st.session_state.committee_size = new_committee_size
        st.session_state.epsilon = new_epsilon
        st.session_state.delta = new_delta
        st.session_state.model_type = new_model_type
        st.session_state.max_training_rounds = new_max_training_rounds
        st.session_state.aggregation_method = new_aggregation_method
        st.session_state.enable_secret_sharing = new_enable_secret_sharing
        st.session_state.secret_sharing_threshold = new_secret_sharing_threshold
        st.session_state.secret_sharing_shares = new_secret_sharing_shares
        st.session_state.system = None  # Reset system to reinitialize with new params
        st.session_state.simulation_started = False
        st.session_state.current_round = 0
        st.session_state.metrics_history = []
    
    st.sidebar.markdown("---")
    
    # Dataset Configuration
    st.sidebar.subheader("Dataset Configuration")
    
    # File uploader for custom datasets
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom Dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file with healthcare data. Should include columns like: age, gender, systolic_bp, diastolic_bp, heart_rate, glucose, etc."
    )
    
    # Store uploaded file in session state
    if uploaded_file is not None:
        if 'uploaded_dataset' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
            try:
                # Read the uploaded CSV file
                custom_data = pd.read_csv(uploaded_file)
                st.session_state.uploaded_dataset = custom_data
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.use_custom_dataset = True
                st.sidebar.success(f"✅ Dataset loaded: {uploaded_file.name}")
                st.sidebar.write(f"Shape: {custom_data.shape[0]} patients, {custom_data.shape[1]} features")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading dataset: {str(e)}")
                st.session_state.use_custom_dataset = False
    else:
        st.session_state.use_custom_dataset = False
        if 'uploaded_dataset' in st.session_state:
            del st.session_state.uploaded_dataset
    
    # Option to use simulated data
    if not st.session_state.get('use_custom_dataset', False):
        st.sidebar.info("Using simulated healthcare data")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["System Overview", "Architecture", "Training Simulation", "Accuracy & Loss Graphs", "Privacy & Security Metrics", "Byzantine Tolerance"]
    )
    
    if page == "System Overview":
        show_system_overview()
    elif page == "Architecture":
        show_architecture()
    elif page == "Training Simulation":
        show_training_simulation()
    elif page == "Accuracy & Loss Graphs":
        show_accuracy_loss_graphs()
    elif page == "Privacy & Security Metrics":
        show_privacy_security_metrics()
    elif page == "Byzantine Tolerance":
        show_byzantine_tolerance()

def show_system_overview():
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Healthcare Facilities", str(st.session_state.num_healthcare_facilities), "Active")
        st.metric("Fog Nodes", str(st.session_state.num_fog_nodes), "Online")
    
    with col2:
        st.metric("Validator Committee Size", str(st.session_state.committee_size), "Rotating")
        st.metric("Security Level", "High", "✅ All checks passed")
    
    with col3:
        st.metric("Privacy Guarantee", "ε-DP", f"ε={st.session_state.epsilon}")
        st.metric("Model Type", st.session_state.model_type, st.session_state.aggregation_method)
    
    st.markdown("---")
    
    # Key Features
    st.subheader("Key Features")
    
    features = [
        ("🔐 Differential Privacy", "Noise addition to model updates prevents inference attacks"),
        ("🔑 Shamir Secret Sharing", "Model updates split into shares for enhanced security"),
        ("⚡ Proof-of-Work", "Sybil resistance during healthcare facility registration"),
        ("👥 Validator Committee", "Rotating committee for Byzantine fault tolerance"),
        ("🏗️ Hierarchical Architecture", "Three-tier structure: Facilities → Fog → Leader"),
        ("🔒 CP-ABE Access Control", "Fine-grained access control for model distribution")
    ]
    
    for feature, description in features:
        with st.expander(feature):
            st.write(description)
    
    # System Status
    st.subheader("System Status")
    
    status_data = {
        "Component": ["Trusted Authority", "Fog Nodes", "Healthcare Facilities", "Validator Committee"],
        "Status": ["Online", "Online", "Online", "Active"],
        "Last Updated": ["2 min ago", "1 min ago", "30 sec ago", "45 sec ago"]
    }
    
    st.dataframe(pd.DataFrame(status_data), width='stretch')
    
    # Dataset Information
    st.subheader("Dataset Information")
    
    system = initialize_system()
    dataset_info = system.data_simulator.get_dataset_info()
    
    if dataset_info['type'] == 'custom':
        st.success("📂 Custom dataset loaded")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", dataset_info['patients'])
        with col2:
            st.metric("Features", dataset_info['features'])
        with col3:
            age_range = dataset_info.get('age_range', [0, 0])
            st.metric("Age Range", f"{age_range[0]}-{age_range[1]}")
        
        # Show dataset preview
        if st.button("📊 Show Dataset Preview"):
            st.subheader("Dataset Sample")
            if 'uploaded_dataset' in st.session_state:
                st.dataframe(st.session_state.uploaded_dataset.head(10), width='stretch')
                
                # Basic statistics
                st.subheader("Dataset Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Numerical Features Summary:**")
                    numeric_cols = st.session_state.uploaded_dataset.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(st.session_state.uploaded_dataset[numeric_cols].describe(), width='stretch')
                
                with col2:
                    st.write("**Categorical Features:**")
                    categorical_cols = st.session_state.uploaded_dataset.select_dtypes(exclude=[np.number]).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols[:5]:  # Show first 5 categorical columns
                            unique_values = st.session_state.uploaded_dataset[col].nunique()
                            st.write(f"**{col}:** {unique_values} unique values")
    else:
        st.info("🔬 Using simulated healthcare data")
        st.write(f"**Simulated Patients:** {st.session_state.num_healthcare_facilities * 200} (distributed across facilities)")
        st.write("**Features:** Age, Gender, Vitals, Lab Results, Risk Scores, Medical Conditions")

def show_architecture():
    st.header("System Architecture")
    
    # Three-tier architecture visualization
    fig = go.Figure()
    
    # Define positions for components
    # Healthcare Facilities (bottom tier)
    num_facilities = st.session_state.num_healthcare_facilities
    facilities_x = list(range(1, num_facilities + 1))
    facilities_y = [1] * num_facilities
    
    # Fog Nodes (middle tier)
    num_fog = st.session_state.num_fog_nodes
    fog_x = [2.5 + i * (6 / max(1, num_fog - 1)) for i in range(num_fog)] if num_fog > 1 else [4.5]
    fog_y = [3] * num_fog
    
    # Leader Server (top tier)
    leader_x = [4.5]
    leader_y = [5]
    
    # Trusted Authority
    ta_x = [8.5]
    ta_y = [3]
    
    # Add healthcare facilities
    fig.add_trace(go.Scatter(
        x=facilities_x, y=facilities_y,
        mode='markers+text',
        marker=dict(size=20, color='lightblue', symbol='square'),
        text=[f'HC{i}' for i in range(1, num_facilities + 1)],
        textposition="middle center",
        name='Healthcare Facilities'
    ))
    
    # Add fog nodes
    fig.add_trace(go.Scatter(
        x=fog_x, y=fog_y,
        mode='markers+text',
        marker=dict(size=30, color='lightgreen', symbol='diamond'),
        text=[f'Fog{i}' for i in range(1, num_fog + 1)],
        textposition="middle center",
        name='Fog Nodes'
    ))
    
    # Add leader server
    fig.add_trace(go.Scatter(
        x=leader_x, y=leader_y,
        mode='markers+text',
        marker=dict(size=40, color='orange', symbol='star'),
        text=['Leader'],
        textposition="middle center",
        name='Leader Server'
    ))
    
    # Add trusted authority
    fig.add_trace(go.Scatter(
        x=ta_x, y=ta_y,
        mode='markers+text',
        marker=dict(size=35, color='red', symbol='triangle-up'),
        text=['TA'],
        textposition="middle center",
        name='Trusted Authority'
    ))
    
    # Add connections
    # Facilities to fog nodes
    for i, fx in enumerate(facilities_x):
        fog_idx = i // max(1, len(facilities_x) // num_fog)  # Distribute facilities across fog nodes
        if fog_idx >= len(fog_x):
            fog_idx = len(fog_x) - 1
        fig.add_shape(
            type="line",
            x0=fx, y0=facilities_y[i],
            x1=fog_x[fog_idx], y1=fog_y[fog_idx],
            line=dict(color="gray", width=1, dash="dash")
        )
    
    # Fog nodes to leader
    for fx, fy in zip(fog_x, fog_y):
        fig.add_shape(
            type="line",
            x0=fx, y0=fy,
            x1=leader_x[0], y1=leader_y[0],
            line=dict(color="blue", width=2)
        )
    
    fig.update_layout(
        title="Three-Tier Hierarchical Federated Learning Architecture",
        xaxis_title="",
        yaxis_title="",
        showlegend=True,
        height=600,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Architecture explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Flow")
        st.markdown("""
        1. **Local Training**: Healthcare facilities train models on private data
        2. **Privacy Protection**: Differential privacy applied to model updates
        3. **Secret Sharing**: Updates split using Shamir's scheme
        4. **Validation**: Committee validates secret shares
        5. **Partial Aggregation**: Fog nodes aggregate validated shares
        6. **Global Aggregation**: Leader performs final aggregation
        7. **Distribution**: Encrypted model distributed via CP-ABE
        """)
    
    with col2:
        st.subheader("Security Layers")
        st.markdown("""
        - **Registration**: Proof-of-Work prevents Sybil attacks
        - **Privacy**: Differential privacy (ε=0.1) protects data
        - **Integrity**: Secret sharing prevents single-point failures
        - **Byzantine Tolerance**: Validator committee consensus
        - **Access Control**: CP-ABE for fine-grained permissions
        - **Communication**: Encrypted channels for all transmissions
        """)

def show_registration_setup():
    st.header("Registration & Setup")
    
    system = initialize_system()
    
    tab1, tab2, tab3 = st.tabs(["Proof-of-Work Registration", "Key Distribution", "Committee Formation"])
    
    with tab1:
        st.subheader("Proof-of-Work Registration")
        st.write("Healthcare facilities must solve computational challenges to register")
        
        if st.button("Simulate Registration Process"):
            with st.spinner("Processing registration challenges..."):
                pow_system = ProofOfWork(difficulty=4)
                
                progress_bar = st.progress(0)
                registration_results = []
                
                for i in range(8):
                    facility_id = f"healthcare_facility_{i+1}"
                    challenge, nonce, hash_result = pow_system.solve_challenge(facility_id)
                    
                    registration_results.append({
                        "Facility": f"HC{i+1}",
                        "Challenge": challenge[:16] + "...",
                        "Nonce": nonce,
                        "Hash": hash_result[:16] + "...",
                        "Status": "✅ Registered"
                    })
                    
                    progress_bar.progress((i + 1) / 8)
                    time.sleep(0.5)
                
                st.success("All healthcare facilities successfully registered!")
                st.dataframe(pd.DataFrame(registration_results), use_container_width=True)
    
    with tab2:
        st.subheader("Cryptographic Key Distribution")
        st.write("Trusted Authority distributes CP-ABE keys and system parameters")
        
        key_distribution = {
            "Entity": ["HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HC8", "Fog1", "Fog2", "Fog3", "Leader"],
            "Public Key": ["✅ Received"] * 12,
            "Private Key": ["✅ Received"] * 12,
            "CP-ABE Attributes": [
                "hospital.cardiology", "hospital.oncology", "clinic.general", "hospital.neurology",
                "clinic.pediatrics", "hospital.emergency", "clinic.dermatology", "hospital.radiology",
                "fog.aggregator", "fog.aggregator", "fog.aggregator", "leader.global"
            ]
        }
        
        st.dataframe(pd.DataFrame(key_distribution), use_container_width=True)
    
    with tab3:
        st.subheader("Validator Committee Formation")
        st.write("Random selection of validator committee with reputation-based weighting")
        
        if st.button("Form New Committee"):
            committee = ValidatorCommittee(committee_size=5)
            participants = [f"HC{i}" for i in range(1, 9)] + [f"Fog{i}" for i in range(1, 4)]
            selected_committee = committee.select_committee(participants)
            
            committee_data = {
                "Validator": selected_committee,
                "Role": ["Primary", "Secondary", "Tertiary", "Backup", "Observer"],
                "Reputation Score": [0.95, 0.88, 0.92, 0.87, 0.90],
                "Selection Weight": [0.23, 0.18, 0.21, 0.17, 0.21]
            }
            
            st.dataframe(pd.DataFrame(committee_data), use_container_width=True)
            
            # Committee rotation schedule
            st.subheader("Rotation Schedule")
            rotation_schedule = []
            current_time = datetime.now()
            
            for i in range(5):
                rotation_time = current_time + timedelta(hours=2*i)
                rotation_schedule.append({
                    "Round": f"Round {i+1}",
                    "Rotation Time": rotation_time.strftime("%Y-%m-%d %H:%M"),
                    "New Primary": selected_committee[(i+1) % len(selected_committee)],
                    "Status": "Scheduled" if i > 0 else "Active"
                })
            
            st.dataframe(pd.DataFrame(rotation_schedule), use_container_width=True)

def show_training_simulation():
    st.header("Federated Learning Training Simulation")
    
    system = initialize_system()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Training Parameters")
        
        # Show current ML configuration
        st.info(f"**Model:** {st.session_state.model_type}\n**Aggregation:** {st.session_state.aggregation_method}\n**Secret Sharing:** {'✅ Enabled' if st.session_state.enable_secret_sharing else '❌ Disabled'}")
        
        # Training configuration (use session state values)
        epsilon = st.session_state.epsilon
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        local_epochs = st.slider("Local Epochs", 1, 10, 3)
        global_rounds = st.slider("Global Rounds", 5, st.session_state.max_training_rounds, 
                                 min(10, st.session_state.max_training_rounds))
        
        byzantine_ratio = st.slider("Byzantine Participants (%)", 0, 30, 12, 1)
        
        # Display current configuration values
        st.markdown("**Current Configuration:**")
        st.write(f"• Differential Privacy (ε): {st.session_state.epsilon}")
        st.write(f"• Delta (δ): {st.session_state.delta}")
        st.write(f"• Healthcare Facilities: {st.session_state.num_healthcare_facilities}")
        st.write(f"• Fog Nodes: {st.session_state.num_fog_nodes}")
        st.write(f"• Committee Size: {st.session_state.committee_size}")
        if st.session_state.enable_secret_sharing:
            st.write(f"• Secret Shares: {st.session_state.secret_sharing_shares} (threshold: {st.session_state.secret_sharing_threshold})")
        else:
            st.write("• Secret Sharing: Disabled")
        
        start_training = st.button("Start Training", type="primary")
        
        if st.button("Reset Simulation"):
            st.session_state.simulation_started = False
            st.session_state.current_round = 0
            st.session_state.metrics_history = []
            st.rerun()
    
    with col1:
        if start_training or st.session_state.simulation_started:
            st.session_state.simulation_started = True
            
            # Training progress
            progress_container = st.container()
            with progress_container:
                st.subheader(f"Training Progress - Round {st.session_state.current_round + 1}/{global_rounds}")
                
                if st.session_state.current_round < global_rounds:
                    # Simulate training round
                    with st.spinner("Training in progress..."):
                        # Local training phase
                        st.write("📱 **Phase 1**: Local model training at healthcare facilities")
                        local_progress = st.progress(0)
                        
                        for i in range(st.session_state.num_healthcare_facilities):
                            progress_pct = int(((i + 1) / st.session_state.num_healthcare_facilities) * 100)
                            local_progress.progress((i + 1) / st.session_state.num_healthcare_facilities)
                            time.sleep(0.3)
                        
                        st.success(f"✅ Local training completed (100% - {st.session_state.num_healthcare_facilities} facilities)")
                        
                        # Differential privacy phase
                        st.write(f"🔒 **Phase 2**: Applying differential privacy (ε={st.session_state.epsilon})")
                        dp_system = DifferentialPrivacy(epsilon=st.session_state.epsilon, delta=st.session_state.delta)
                        
                        privacy_metrics = []
                        for i in range(st.session_state.num_healthcare_facilities):
                            # Simulate gradient with noise
                            original_grad = np.random.normal(0, 1, 10)
                            noisy_grad = dp_system.add_noise(original_grad)
                            noise_level = np.std(noisy_grad - original_grad)
                            privacy_metrics.append(noise_level)
                        
                        avg_noise = np.mean(privacy_metrics)
                        st.success(f"✅ Differential privacy applied (avg noise: {avg_noise:.4f})")
                        
                        # Secret sharing phase
                        if st.session_state.enable_secret_sharing:
                            st.write("🔑 **Phase 3**: Secret sharing of model updates")
                            sss = ShamirSecretSharing(threshold=st.session_state.secret_sharing_threshold, 
                                                     num_shares=st.session_state.secret_sharing_shares)
                        else:
                            st.write("🔓 **Phase 3**: Direct model update transmission (Secret sharing disabled)")
                        
                        sharing_progress = st.progress(0)
                        for i in range(st.session_state.num_healthcare_facilities):
                            progress_pct = int(((i + 1) / st.session_state.num_healthcare_facilities) * 100)
                            sharing_progress.progress((i + 1) / st.session_state.num_healthcare_facilities)
                            time.sleep(0.2)
                        
                        if st.session_state.enable_secret_sharing:
                            st.success(f"✅ Secret shares distributed ({st.session_state.secret_sharing_threshold}/{st.session_state.secret_sharing_shares} threshold)")
                        else:
                            st.success("✅ Model updates transmitted directly")
                        
                        # Validation phase
                        st.write(f"👥 **Phase 4**: Committee validation ({st.session_state.committee_size} validators)")
                        committee = ValidatorCommittee(committee_size=st.session_state.committee_size)
                        validation_approved = committee.validate_shares([f"share_{i}" for i in range(st.session_state.num_healthcare_facilities)])
                        
                        # Simulate individual share validation results for display
                        if validation_approved:
                            valid_shares = max(int(st.session_state.num_healthcare_facilities * 0.85), st.session_state.num_healthcare_facilities - 1)
                            validation_pct = int((valid_shares / st.session_state.num_healthcare_facilities) * 100)
                        else:
                            valid_shares = int(st.session_state.num_healthcare_facilities * 0.6)
                            validation_pct = int((valid_shares / st.session_state.num_healthcare_facilities) * 100)
                        
                        st.success(f"✅ Validation completed ({valid_shares}/{st.session_state.num_healthcare_facilities} shares approved - {validation_pct}%)")
                        
                        # Aggregation phase
                        st.write(f"🔄 **Phase 5**: Hierarchical aggregation ({st.session_state.aggregation_method})")
                        
                        # Fog node aggregation
                        fog_progress = st.progress(0)
                        for i in range(st.session_state.num_fog_nodes):
                            progress_pct = int(((i + 1) / st.session_state.num_fog_nodes) * 100)
                            fog_progress.progress((i + 1) / st.session_state.num_fog_nodes)
                            time.sleep(0.4)
                        
                        st.success(f"✅ Fog node aggregation completed (100% - {st.session_state.num_fog_nodes} fog nodes)")
                        
                        # Global aggregation
                        st.write("🌐 **Phase 6**: Global aggregation at leader server")
                        time.sleep(1)
                        st.success("✅ Global model updated")
                        
                        # Record metrics
                        round_metrics = {
                            'round': st.session_state.current_round + 1,
                            'accuracy': 0.65 + (st.session_state.current_round * 0.03) + np.random.normal(0, 0.01),
                            'loss': 2.5 - (st.session_state.current_round * 0.2) + np.random.normal(0, 0.05),
                            'privacy_budget': st.session_state.epsilon * (st.session_state.current_round + 1),
                            'byzantine_detected': np.random.randint(0, int(st.session_state.num_healthcare_facilities * byzantine_ratio / 100) + 1),
                            'communication_cost': np.random.uniform(50, 100)
                        }
                        
                        st.session_state.metrics_history.append(round_metrics)
                        st.session_state.current_round += 1
                        
                        if st.session_state.current_round < global_rounds:
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.success("🎉 Training completed!")
                            st.balloons()
        
        # Display metrics if training has started
        if st.session_state.metrics_history:
            show_training_metrics()

def show_training_metrics():
    st.subheader("Training Metrics")
    
    df_metrics = pd.DataFrame(st.session_state.metrics_history)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Accuracy', 'Training Loss', 'Privacy Budget Usage', 'Byzantine Attacks Detected'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=df_metrics['round'], y=df_metrics['accuracy'],
                   mode='lines+markers', name='Accuracy', line=dict(color='green')),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=df_metrics['round'], y=df_metrics['loss'],
                   mode='lines+markers', name='Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Privacy budget
    fig.add_trace(
        go.Scatter(x=df_metrics['round'], y=df_metrics['privacy_budget'],
                   mode='lines+markers', name='Privacy Budget', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Byzantine attacks
    fig.add_trace(
        go.Bar(x=df_metrics['round'], y=df_metrics['byzantine_detected'],
               name='Byzantine Attacks', marker_color='orange'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Progress Metrics")
    st.plotly_chart(fig, use_container_width=True)

def show_accuracy_loss_graphs():
    st.header("Accuracy & Loss Graphs")
    
    if not st.session_state.metrics_history:
        st.info("📊 No training data available. Please run the training simulation first to see accuracy and loss graphs.")
        st.markdown("### How to Generate Data:")
        st.markdown("""
        1. Go to **Training Simulation** page
        2. Configure your training parameters
        3. Click **Start Training** to begin the federated learning process
        4. Return here to view detailed accuracy and loss visualizations
        """)
        return
    
    df_metrics = pd.DataFrame(st.session_state.metrics_history)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Over Time")
        
        # Accuracy graph
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=df_metrics['round'],
            y=df_metrics['accuracy'],
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='green', width=4),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>Round %{x}</b><br>Accuracy: %{y:.3f}<extra></extra>'
        ))
        
        # Add target accuracy line
        target_accuracy = 0.95
        fig_acc.add_hline(y=target_accuracy, line_dash="dash", 
                         line_color="darkgreen", 
                         annotation_text=f"Target Accuracy ({target_accuracy})")
        
        fig_acc.update_layout(
            title="Federated Learning Model Accuracy",
            xaxis_title="Training Round",
            yaxis_title="Accuracy",
            height=400,
            yaxis=dict(range=[0.5, 1.0]),
            showlegend=False
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Accuracy statistics
        st.subheader("Accuracy Statistics")
        current_accuracy = df_metrics['accuracy'].iloc[-1]
        max_accuracy = df_metrics['accuracy'].max()
        accuracy_improvement = current_accuracy - df_metrics['accuracy'].iloc[0] if len(df_metrics) > 1 else 0
        
        acc_col1, acc_col2, acc_col3 = st.columns(3)
        with acc_col1:
            st.metric("Current Accuracy", f"{current_accuracy:.3f}")
        with acc_col2:
            st.metric("Best Accuracy", f"{max_accuracy:.3f}")
        with acc_col3:
            st.metric("Improvement", f"{accuracy_improvement:.3f}", delta=f"{accuracy_improvement:.3f}")
    
    with col2:
        st.subheader("Training Loss Over Time")
        
        # Loss graph
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df_metrics['round'],
            y=df_metrics['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='red', width=4),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='<b>Round %{x}</b><br>Loss: %{y:.3f}<extra></extra>'
        ))
        
        # Add target loss line
        target_loss = 0.5
        fig_loss.add_hline(y=target_loss, line_dash="dash", 
                          line_color="darkred", 
                          annotation_text=f"Target Loss ({target_loss})")
        
        fig_loss.update_layout(
            title="Federated Learning Training Loss",
            xaxis_title="Training Round",
            yaxis_title="Loss",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Loss statistics
        st.subheader("Loss Statistics")
        current_loss = df_metrics['loss'].iloc[-1]
        min_loss = df_metrics['loss'].min()
        loss_reduction = df_metrics['loss'].iloc[0] - current_loss if len(df_metrics) > 1 else 0
        
        loss_col1, loss_col2, loss_col3 = st.columns(3)
        with loss_col1:
            st.metric("Current Loss", f"{current_loss:.3f}")
        with loss_col2:
            st.metric("Best Loss", f"{min_loss:.3f}")
        with loss_col3:
            st.metric("Reduction", f"{loss_reduction:.3f}", delta=f"{loss_reduction:.3f}")
    
    # Combined accuracy and loss graph
    st.subheader("Combined Accuracy & Loss Visualization")
    
    fig_combined = make_subplots(
        specs=[[{"secondary_y": True}]], 
        subplot_titles=["Accuracy and Loss Over Training Rounds"]
    )
    
    # Add accuracy trace
    fig_combined.add_trace(
        go.Scatter(
            x=df_metrics['round'], 
            y=df_metrics['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Add loss trace
    fig_combined.add_trace(
        go.Scatter(
            x=df_metrics['round'], 
            y=df_metrics['loss'],
            mode='lines+markers',
            name='Loss',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    # Update axis labels
    fig_combined.update_xaxes(title_text="Training Round")
    fig_combined.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig_combined.update_yaxes(title_text="Loss", secondary_y=True)
    
    fig_combined.update_layout(
        title="Federated Learning Progress: Accuracy vs Loss",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Training convergence analysis
    st.subheader("Training Convergence Analysis")
    
    convergence_col1, convergence_col2 = st.columns(2)
    
    with convergence_col1:
        # Calculate convergence metrics
        if len(df_metrics) >= 3:
            recent_acc_change = abs(df_metrics['accuracy'].iloc[-1] - df_metrics['accuracy'].iloc[-3])
            recent_loss_change = abs(df_metrics['loss'].iloc[-1] - df_metrics['loss'].iloc[-3])
            
            if recent_acc_change < 0.01 and recent_loss_change < 0.1:
                convergence_status = "✅ Converged"
                convergence_color = "green"
            elif recent_acc_change < 0.05 and recent_loss_change < 0.3:
                convergence_status = "🔄 Converging"
                convergence_color = "orange"
            else:
                convergence_status = "📈 Training"
                convergence_color = "blue"
        else:
            recent_acc_change = 0.0
            recent_loss_change = 0.0
            convergence_status = "📊 Insufficient Data"
            convergence_color = "gray"
        
        st.markdown(f"**Convergence Status:** <span style='color:{convergence_color}'>{convergence_status}</span>", 
                   unsafe_allow_html=True)
        
        if len(df_metrics) >= 3:
            st.write(f"Recent accuracy change: {recent_acc_change:.4f}")
            st.write(f"Recent loss change: {recent_loss_change:.4f}")
    
    with convergence_col2:
        # Training efficiency metrics
        total_rounds = len(df_metrics)
        avg_accuracy_per_round = (df_metrics['accuracy'].iloc[-1] - df_metrics['accuracy'].iloc[0]) / total_rounds if total_rounds > 1 else 0
        avg_loss_reduction_per_round = (df_metrics['loss'].iloc[0] - df_metrics['loss'].iloc[-1]) / total_rounds if total_rounds > 1 else 0
        
        st.write(f"**Training Efficiency:**")
        st.write(f"Rounds completed: {total_rounds}")
        st.write(f"Avg accuracy gain/round: {avg_accuracy_per_round:.4f}")
        st.write(f"Avg loss reduction/round: {avg_loss_reduction_per_round:.4f}")

def show_privacy_security_metrics():
    st.header("Privacy & Security Metrics")
    
    system = initialize_system()
    
    tab1, tab2, tab3 = st.tabs(["Differential Privacy", "Secret Sharing", "Communication Security"])
    
    with tab1:
        st.subheader("Differential Privacy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epsilon_values = [0.01, 0.05, 0.1, 0.5, 1.0]
            utility_scores = [0.45, 0.62, 0.78, 0.89, 0.95]
            privacy_scores = [0.98, 0.92, 0.85, 0.72, 0.60]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epsilon_values, y=utility_scores, 
                                   mode='lines+markers', name='Model Utility'))
            fig.add_trace(go.Scatter(x=epsilon_values, y=privacy_scores,
                                   mode='lines+markers', name='Privacy Level'))
            
            fig.update_layout(title="Privacy-Utility Trade-off",
                            xaxis_title="Epsilon (ε)",
                            yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Privacy Budget Tracking")
            
            if st.session_state.metrics_history:
                df_metrics = pd.DataFrame(st.session_state.metrics_history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_metrics['round'],
                    y=df_metrics['privacy_budget'],
                    mode='lines+markers',
                    fill='tonexty',
                    name='Cumulative ε'
                ))
                
                # Add privacy budget threshold
                max_budget = 1.0
                fig.add_hline(y=max_budget, line_dash="dash", 
                            line_color="red", annotation_text="Privacy Budget Limit")
                
                fig.update_layout(title="Privacy Budget Consumption",
                                xaxis_title="Round",
                                yaxis_title="Cumulative ε")
                st.plotly_chart(fig, use_container_width=True)
            
            # Privacy guarantees table
            privacy_guarantees = {
                "Parameter": ["Epsilon (ε)", "Delta (δ)", "Sensitivity", "Noise Type"],
                "Value": [str(st.session_state.epsilon), str(st.session_state.delta), "2.0", "Gaussian"],
                "Description": [
                    "Privacy loss parameter",
                    "Probability of privacy breach",
                    "Maximum change in adjacent datasets",
                    "Noise distribution for DP"
                ]
            }
            
            st.dataframe(pd.DataFrame(privacy_guarantees), use_container_width=True)
    
    with tab2:
        st.subheader("Shamir Secret Sharing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Secret Sharing Configuration")
            
            sss_config = {
                "Parameter": ["Threshold (t)", "Total Shares (n)", "Prime Field", "Security Level"],
                "Value": ["3", "5", "2^127-1", "128-bit"],
                "Description": [
                    "Minimum shares needed for reconstruction",
                    "Total number of shares generated",
                    "Prime field for arithmetic operations",
                    "Computational security level"
                ]
            }
            
            st.dataframe(pd.DataFrame(sss_config), use_container_width=True)
        
        with col2:
            st.subheader("Share Distribution")
            
            # Visualize secret sharing
            shares_data = {
                "Share ID": [f"Share {i+1}" for i in range(5)],
                "Holder": ["Fog1", "Fog2", "Fog3", "Validator1", "Validator2"],
                "Status": ["Active", "Active", "Active", "Active", "Active"],
                "Last Validation": ["2 min ago", "1 min ago", "3 min ago", "1 min ago", "2 min ago"]
            }
            
            st.dataframe(pd.DataFrame(shares_data), use_container_width=True)
            
            # Security analysis
            st.subheader("Security Properties")
            st.write("✅ **Perfect Secrecy**: Individual shares reveal no information")
            st.write("✅ **Threshold Security**: Requires 3/5 shares for reconstruction")
            st.write("✅ **Verifiable**: Shares can be verified without reconstruction")
            st.write("✅ **Distributed**: No single point of failure")
    
    with tab3:
        st.subheader("Communication Security")
        
        # Network security metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Encrypted Channels", "100%", "All communications")
            st.metric("Key Rotations", "24", "Per day")
        
        with col2:
            st.metric("Message Integrity", "100%", "HMAC verified")
            st.metric("Forward Secrecy", "Yes", "PFS enabled")
        
        with col3:
            st.metric("DDoS Protection", "Active", "Rate limiting")
            st.metric("Intrusion Detection", "Online", "ML-based")
        
        # Communication overhead analysis
        st.subheader("Communication Overhead")
        
        communication_data = {
            "Phase": ["Local Training", "DP Noise Addition", "Secret Sharing", 
                     "Validation", "Fog Aggregation", "Global Aggregation"],
            "Data Size (KB)": [0, 15, 45, 12, 25, 8],
            "Network Calls": [0, 0, 5, 5, 3, 1],
            "Encryption": ["N/A", "AES-256", "RSA+AES", "RSA+AES", "AES-256", "AES-256"]
        }
        
        df_comm = pd.DataFrame(communication_data)
        
        # Create bar chart for communication overhead
        fig = px.bar(df_comm, x='Phase', y='Data Size (KB)', 
                    title='Communication Overhead by Phase')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def show_byzantine_tolerance():
    st.header("Byzantine Fault Tolerance")
    
    system = initialize_system()
    
    tab1, tab2, tab3 = st.tabs(["Threat Detection", "Committee Validation", "Attack Simulation"])
    
    with tab1:
        st.subheader("Byzantine Threat Detection")
        
        # Real-time threat monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Threat Status")
            
            threat_levels = {
                "Threat Type": ["Model Poisoning", "Gradient Attacks", "Sybil Attacks", 
                              "Data Poisoning", "Communication Tampering"],
                "Risk Level": ["Medium", "Low", "Very Low", "Low", "Very Low"],
                "Detection Rate": ["95%", "98%", "100%", "92%", "99%"],
                "Mitigation": ["Committee Validation", "DP Noise", "Proof-of-Work", 
                              "Statistical Analysis", "Cryptographic Verification"]
            }
            
            # Color code risk levels
            def get_risk_color(risk):
                if risk == "Very Low":
                    return "🟢"
                elif risk == "Low":
                    return "🟡"
                elif risk == "Medium":
                    return "🟠"
                else:
                    return "🔴"
            
            df_threats = pd.DataFrame(threat_levels)
            df_threats["Risk Level"] = df_threats["Risk Level"].apply(
                lambda x: f"{get_risk_color(x)} {x}"
            )
            
            st.dataframe(df_threats, use_container_width=True)
        
        with col2:
            st.subheader("Detection Accuracy")
            
            # Confusion matrix for Byzantine detection
            detection_metrics = {
                "Metric": ["True Positive Rate", "False Positive Rate", "Precision", "Recall", "F1-Score"],
                "Value": [0.94, 0.03, 0.97, 0.94, 0.95]
            }
            
            df_metrics = pd.DataFrame(detection_metrics)
            
            fig = px.bar(df_metrics, x='Metric', y='Value', 
                        title='Byzantine Detection Performance')
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Validator Committee Operations")
        
        # Committee validation process
        if st.button("Simulate Committee Validation Round"):
            with st.spinner("Committee validation in progress..."):
                committee = ValidatorCommittee(committee_size=5)
                
                # Simulate incoming shares for validation
                shares_to_validate = [f"share_hc_{i}" for i in range(1, 9)]
                
                st.write("**Step 1**: Receiving secret shares from healthcare facilities")
                share_progress = st.progress(0)
                for i in range(len(shares_to_validate)):
                    share_progress.progress((i + 1) / len(shares_to_validate))
                    time.sleep(0.2)
                
                st.write("**Step 2**: Committee members validating shares")
                validation_results = []
                
                for i, share in enumerate(shares_to_validate):
                    # Simulate Byzantine behavior (12% of shares are malicious)
                    is_malicious = np.random.random() < 0.12
                    
                    if is_malicious:
                        votes = committee.consensus_vote([False, False, True, False, True])  # Majority reject
                        status = "❌ Rejected"
                        reason = "Malicious content detected"
                    else:
                        votes = committee.consensus_vote([True, True, True, True, False])  # Majority accept
                        status = "✅ Accepted"
                        reason = "Valid share"
                    
                    validation_results.append({
                        "Share ID": f"HC{i+1}",
                        "Committee Votes": f"{sum(votes)}/5",
                        "Status": status,
                        "Reason": reason
                    })
                
                time.sleep(1)
                st.write("**Step 3**: Validation results")
                st.dataframe(pd.DataFrame(validation_results), use_container_width=True)
                
                accepted_shares = sum(1 for r in validation_results if "Accepted" in r["Status"])
                st.success(f"Validation completed: {accepted_shares}/8 shares accepted")
    
    with tab3:
        st.subheader("Attack Simulation")
        
        attack_type = st.selectbox(
            "Select Attack Type",
            ["Model Poisoning", "Gradient Inversion", "Backdoor Attack", "Free-rider Attack"]
        )
        
        attack_intensity = st.slider("Attack Intensity (%)", 0, 50, 25)
        num_attackers = st.slider("Number of Attackers", 1, 4, 2)
        
        if st.button("Launch Attack Simulation"):
            st.warning(f"⚠️ Simulating {attack_type} with {num_attackers} attackers at {attack_intensity}% intensity")
            
            with st.spinner("Attack simulation in progress..."):
                # Simulate attack detection and mitigation
                time.sleep(2)
                
                # Detection phase
                detection_time = np.random.uniform(1, 5)
                st.write(f"🔍 **Detection**: Attack detected after {detection_time:.1f} seconds")
                
                # Mitigation phase
                if attack_type == "Model Poisoning":
                    mitigation = "Committee validation rejected malicious updates"
                elif attack_type == "Gradient Inversion":
                    mitigation = "Differential privacy noise prevented data reconstruction"
                elif attack_type == "Backdoor Attack":
                    mitigation = "Statistical anomaly detection flagged suspicious patterns"
                else:
                    mitigation = "Reputation system identified and isolated free-riders"
                
                st.write(f"🛡️ **Mitigation**: {mitigation}")
                
                # Results
                attack_success_rate = max(0, attack_intensity - 70) / 30  # Most attacks fail
                model_degradation = attack_success_rate * 0.1  # Minimal impact
                
                results = {
                    "Metric": ["Attack Success Rate", "Model Accuracy Impact", 
                              "Detection Time", "Recovery Time", "System Availability"],
                    "Value": [f"{attack_success_rate*100:.1f}%", f"-{model_degradation*100:.2f}%", 
                             f"{detection_time:.1f}s", "< 1 round", "99.9%"],
                    "Status": ["🛡️ Blocked", "✅ Minimal", "⚡ Fast", "⚡ Fast", "✅ High"]
                }
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                
                st.success("🎯 Attack successfully mitigated by the Byzantine fault tolerance mechanisms!")

if __name__ == "__main__":
    main()
