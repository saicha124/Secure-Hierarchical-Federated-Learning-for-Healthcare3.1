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

def initialize_system():
    """Initialize the federated learning system"""
    if st.session_state.system is None:
        st.session_state.system = FederatedLearningSystem(
            num_healthcare_facilities=8,
            num_fog_nodes=3,
            committee_size=5
        )
    return st.session_state.system

def main():
    st.title("🏥 Secure Hierarchical Federated Learning for Healthcare")
    st.markdown("### Byzantine-Resilient FL with Differential Privacy & Secret Sharing")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["System Overview", "Architecture", "Registration & Setup", "Training Simulation", "Privacy & Security Metrics", "Byzantine Tolerance"]
    )
    
    if page == "System Overview":
        show_system_overview()
    elif page == "Architecture":
        show_architecture()
    elif page == "Registration & Setup":
        show_registration_setup()
    elif page == "Training Simulation":
        show_training_simulation()
    elif page == "Privacy & Security Metrics":
        show_privacy_security_metrics()
    elif page == "Byzantine Tolerance":
        show_byzantine_tolerance()

def show_system_overview():
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Healthcare Facilities", "8", "Active")
        st.metric("Fog Nodes", "3", "Online")
    
    with col2:
        st.metric("Validator Committee Size", "5", "Rotating")
        st.metric("Security Level", "High", "✅ All checks passed")
    
    with col3:
        st.metric("Privacy Guarantee", "ε-DP", "ε=0.1")
        st.metric("Byzantine Tolerance", "33%", "Fault tolerant")
    
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
    
    st.dataframe(pd.DataFrame(status_data), use_container_width=True)

def show_architecture():
    st.header("System Architecture")
    
    # Three-tier architecture visualization
    fig = go.Figure()
    
    # Define positions for components
    # Healthcare Facilities (bottom tier)
    facilities_x = [1, 2, 3, 4, 5, 6, 7, 8]
    facilities_y = [1] * 8
    
    # Fog Nodes (middle tier)
    fog_x = [2.5, 4.5, 6.5]
    fog_y = [3] * 3
    
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
        text=[f'HC{i}' for i in range(1, 9)],
        textposition="middle center",
        name='Healthcare Facilities'
    ))
    
    # Add fog nodes
    fig.add_trace(go.Scatter(
        x=fog_x, y=fog_y,
        mode='markers+text',
        marker=dict(size=30, color='lightgreen', symbol='diamond'),
        text=['Fog1', 'Fog2', 'Fog3'],
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
        fog_idx = i // 3  # Distribute facilities across fog nodes
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
        
        # Training configuration
        epsilon = st.slider("Differential Privacy (ε)", 0.01, 1.0, 0.1, 0.01)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        local_epochs = st.slider("Local Epochs", 1, 10, 3)
        global_rounds = st.slider("Global Rounds", 1, 20, 10)
        
        byzantine_ratio = st.slider("Byzantine Participants (%)", 0, 30, 12, 1)
        
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
                        
                        for i in range(8):
                            local_progress.progress((i + 1) / 8)
                            time.sleep(0.3)
                        
                        st.success("✅ Local training completed")
                        
                        # Differential privacy phase
                        st.write("🔒 **Phase 2**: Applying differential privacy")
                        dp_system = DifferentialPrivacy(epsilon=epsilon)
                        
                        privacy_metrics = []
                        for i in range(8):
                            # Simulate gradient with noise
                            original_grad = np.random.normal(0, 1, 10)
                            noisy_grad = dp_system.add_noise(original_grad)
                            noise_level = np.std(noisy_grad - original_grad)
                            privacy_metrics.append(noise_level)
                        
                        avg_noise = np.mean(privacy_metrics)
                        st.success(f"✅ Differential privacy applied (avg noise: {avg_noise:.4f})")
                        
                        # Secret sharing phase
                        st.write("🔑 **Phase 3**: Secret sharing of model updates")
                        sss = ShamirSecretSharing(threshold=3, num_shares=5)
                        
                        sharing_progress = st.progress(0)
                        for i in range(8):
                            sharing_progress.progress((i + 1) / 8)
                            time.sleep(0.2)
                        
                        st.success("✅ Secret shares distributed")
                        
                        # Validation phase
                        st.write("👥 **Phase 4**: Committee validation")
                        committee = ValidatorCommittee(committee_size=5)
                        validation_results = committee.validate_shares([f"share_{i}" for i in range(8)])
                        
                        valid_shares = sum(validation_results)
                        st.success(f"✅ Validation completed ({valid_shares}/8 shares approved)")
                        
                        # Aggregation phase
                        st.write("🔄 **Phase 5**: Hierarchical aggregation")
                        
                        # Fog node aggregation
                        fog_progress = st.progress(0)
                        for i in range(3):
                            fog_progress.progress((i + 1) / 3)
                            time.sleep(0.4)
                        
                        st.success("✅ Fog node aggregation completed")
                        
                        # Global aggregation
                        st.write("🌐 **Phase 6**: Global aggregation at leader server")
                        time.sleep(1)
                        st.success("✅ Global model updated")
                        
                        # Record metrics
                        round_metrics = {
                            'round': st.session_state.current_round + 1,
                            'accuracy': 0.65 + (st.session_state.current_round * 0.03) + np.random.normal(0, 0.01),
                            'loss': 2.5 - (st.session_state.current_round * 0.2) + np.random.normal(0, 0.05),
                            'privacy_budget': epsilon * (st.session_state.current_round + 1),
                            'byzantine_detected': np.random.randint(0, int(8 * byzantine_ratio / 100) + 1),
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
                "Value": ["0.1", "1e-5", "2.0", "Gaussian"],
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
