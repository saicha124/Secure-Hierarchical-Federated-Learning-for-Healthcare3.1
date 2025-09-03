import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import networkx as nx
from datetime import datetime, timedelta

class SystemArchitectureViz:
    """Visualization components for system architecture"""
    
    def __init__(self):
        self.colors = {
            'healthcare': '#3498db',  # Blue
            'fog': '#2ecc71',         # Green
            'leader': '#f39c12',      # Orange
            'authority': '#e74c3c',   # Red
            'validator': '#9b59b6'    # Purple
        }
    
    def draw_hierarchical_architecture(self, num_facilities: int = 8, num_fog_nodes: int = 3) -> go.Figure:
        """Draw the three-tier hierarchical architecture"""
        
        fig = go.Figure()
        
        # Define positions for each tier
        # Healthcare Facilities (bottom tier)
        facilities_x = np.linspace(1, 9, num_facilities)
        facilities_y = [1] * num_facilities
        
        # Fog Nodes (middle tier)
        fog_x = np.linspace(2.5, 6.5, num_fog_nodes)
        fog_y = [3] * num_fog_nodes
        
        # Leader Server (top tier)
        leader_x = [4.5]
        leader_y = [5]
        
        # Trusted Authority (side)
        ta_x = [8.5]
        ta_y = [3]
        
        # Add healthcare facilities
        fig.add_trace(go.Scatter(
            x=facilities_x, y=facilities_y,
            mode='markers+text',
            marker=dict(size=25, color=self.colors['healthcare'], symbol='square'),
            text=[f'HC{i}' for i in range(1, num_facilities + 1)],
            textposition="middle center",
            textfont=dict(color='white', size=10),
            name='Healthcare Facilities',
            hovertemplate='<b>%{text}</b><br>Type: Healthcare Facility<br>Status: Active<extra></extra>'
        ))
        
        # Add fog nodes
        fig.add_trace(go.Scatter(
            x=fog_x, y=fog_y,
            mode='markers+text',
            marker=dict(size=35, color=self.colors['fog'], symbol='diamond'),
            text=[f'Fog{i}' for i in range(1, num_fog_nodes + 1)],
            textposition="middle center",
            textfont=dict(color='white', size=12),
            name='Fog Nodes',
            hovertemplate='<b>%{text}</b><br>Type: Fog Node<br>Role: Partial Aggregation<extra></extra>'
        ))
        
        # Add leader server
        fig.add_trace(go.Scatter(
            x=leader_x, y=leader_y,
            mode='markers+text',
            marker=dict(size=45, color=self.colors['leader'], symbol='star'),
            text=['Leader'],
            textposition="middle center",
            textfont=dict(color='white', size=14),
            name='Leader Server',
            hovertemplate='<b>Leader Server</b><br>Type: Global Coordinator<br>Role: Final Aggregation<extra></extra>'
        ))
        
        # Add trusted authority
        fig.add_trace(go.Scatter(
            x=ta_x, y=ta_y,
            mode='markers+text',
            marker=dict(size=40, color=self.colors['authority'], symbol='triangle-up'),
            text=['TA'],
            textposition="middle center",
            textfont=dict(color='white', size=12),
            name='Trusted Authority',
            hovertemplate='<b>Trusted Authority</b><br>Type: System Administrator<br>Role: Key Management & Oversight<extra></extra>'
        ))
        
        # Add connections
        self._add_connections(fig, facilities_x, facilities_y, fog_x, fog_y, leader_x, leader_y)
        
        # Add tier labels
        fig.add_annotation(x=0.5, y=1, text="Tier 1: Healthcare Facilities", 
                          showarrow=False, font=dict(size=14, color='gray'))
        fig.add_annotation(x=0.5, y=3, text="Tier 2: Fog Nodes", 
                          showarrow=False, font=dict(size=14, color='gray'))
        fig.add_annotation(x=0.5, y=5, text="Tier 3: Leader Server", 
                          showarrow=False, font=dict(size=14, color='gray'))
        
        fig.update_layout(
            title="Hierarchical Federated Learning Architecture",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 10]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 6]),
            showlegend=True,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _add_connections(self, fig: go.Figure, facilities_x: np.ndarray, facilities_y: List[float],
                        fog_x: np.ndarray, fog_y: List[float], leader_x: List[float], leader_y: List[float]):
        """Add connection lines between architecture components"""
        
        # Facilities to fog nodes (distribute facilities across fog nodes)
        facilities_per_fog = len(facilities_x) // len(fog_x)
        
        for i, (fx, fy) in enumerate(zip(facilities_x, facilities_y)):
            fog_idx = min(i // max(1, facilities_per_fog), len(fog_x) - 1)
            
            fig.add_shape(
                type="line",
                x0=fx, y0=fy,
                x1=fog_x[fog_idx], y1=fog_y[fog_idx],
                line=dict(color="lightblue", width=2, dash="dot"),
                opacity=0.6
            )
        
        # Fog nodes to leader
        for fx, fy in zip(fog_x, fog_y):
            fig.add_shape(
                type="line",
                x0=fx, y0=fy,
                x1=leader_x[0], y1=leader_y[0],
                line=dict(color="green", width=3),
                opacity=0.8
            )
    
    def draw_data_flow_diagram(self) -> go.Figure:
        """Draw data flow through the system"""
        
        fig = go.Figure()
        
        # Define flow stages
        stages = [
            "Local Training",
            "Differential Privacy",
            "Secret Sharing",
            "Committee Validation",
            "Fog Aggregation",
            "Global Aggregation",
            "Model Distribution"
        ]
        
        # Create flow diagram
        x_positions = list(range(len(stages)))
        y_position = [0] * len(stages)
        
        # Add flow nodes
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_position,
            mode='markers+text',
            marker=dict(size=60, color='lightblue', line=dict(width=2, color='darkblue')),
            text=stages,
            textposition="middle center",
            textfont=dict(size=10),
            name='Data Flow Stages',
            hovertemplate='<b>%{text}</b><br>Stage: %{x}<extra></extra>'
        ))
        
        # Add flow arrows
        for i in range(len(stages) - 1):
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i, ay=0,
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='blue',
                showarrow=True
            )
        
        fig.update_layout(
            title="Data Flow Through Federated Learning Pipeline",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=False,
            height=300,
            plot_bgcolor='white'
        )
        
        return fig
    
    def draw_security_layers(self) -> go.Figure:
        """Visualize security layers in the system"""
        
        security_layers = [
            {"name": "Proof-of-Work", "description": "Sybil Resistance", "level": 1},
            {"name": "Differential Privacy", "description": "Data Protection", "level": 2},
            {"name": "Secret Sharing", "description": "Update Security", "level": 3},
            {"name": "Committee Validation", "description": "Byzantine Tolerance", "level": 4},
            {"name": "CP-ABE", "description": "Access Control", "level": 5}
        ]
        
        fig = go.Figure()
        
        # Create concentric circles for security layers
        theta = np.linspace(0, 2*np.pi, 100)
        
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#cc99ff']
        
        for i, layer in enumerate(security_layers):
            radius = (i + 1) * 0.8
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                fill='tonext' if i > 0 else 'toself',
                fillcolor=colors[i],
                line=dict(color='darkgray', width=2),
                name=f"{layer['name']}: {layer['description']}",
                opacity=0.6,
                hovertemplate=f"<b>{layer['name']}</b><br>{layer['description']}<extra></extra>"
            ))
            
            # Add layer labels
            fig.add_annotation(
                x=radius * 0.7, y=radius * 0.7,
                text=layer['name'],
                showarrow=False,
                font=dict(size=10, color='black')
            )
        
        # Add center point
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=20, color='gold', symbol='star'),
            text=['Healthcare Data'],
            textposition='middle center',
            name='Protected Asset'
        ))
        
        fig.update_layout(
            title="Multi-Layer Security Architecture",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="y"),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=True,
            height=600,
            plot_bgcolor='white'
        )
        
        return fig
    
    def draw_committee_network(self, committee_members: List[str], reputation_scores: Dict[str, float]) -> go.Figure:
        """Visualize validator committee network"""
        
        fig = go.Figure()
        
        # Create network graph for committee
        G = nx.Graph()
        
        # Add nodes (committee members)
        for member in committee_members:
            G.add_node(member, reputation=reputation_scores.get(member, 0.5))
        
        # Add edges (all members connected to each other for consensus)
        for i, member1 in enumerate(committee_members):
            for member2 in committee_members[i+1:]:
                G.add_edge(member1, member2)
        
        # Get positions for network layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_colors = [reputation_scores.get(node, 0.5) for node in G.nodes()]
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_shape(
                type="line",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="lightgray", width=1),
                opacity=0.5
            )
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=30,
                color=node_colors,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Reputation Score"),
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10),
            name='Committee Members',
            hovertemplate='<b>%{text}</b><br>Reputation: %{marker.color:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Validator Committee Network",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        
        return fig


class MetricsViz:
    """Visualization components for system metrics and performance"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_training_progress(self, metrics_history: List[Dict[str, Any]]) -> go.Figure:
        """Plot federated learning training progress"""
        
        if not metrics_history:
            return self._create_empty_plot("No training data available")
        
        df = pd.DataFrame(metrics_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Training Loss', 'Privacy Budget', 'Communication Cost'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(
                x=df['round'], y=df['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                x=df['round'], y=df['loss'],
                mode='lines+markers',
                name='Loss',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Privacy budget
        fig.add_trace(
            go.Scatter(
                x=df['round'], y=df['privacy_budget'],
                mode='lines+markers',
                name='Privacy Budget',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Communication cost
        fig.add_trace(
            go.Scatter(
                x=df['round'], y=df['communication_cost'],
                mode='lines+markers',
                name='Communication Cost',
                line=dict(color='orange', width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Federated Learning Training Metrics"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Round", row=2, col=1)
        fig.update_xaxes(title_text="Round", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative ε", row=2, col=1)
        fig.update_yaxes(title_text="Cost (KB)", row=2, col=2)
        
        return fig
    
    def plot_privacy_metrics(self, epsilon_values: List[float], utility_scores: List[float], 
                           privacy_scores: List[float]) -> go.Figure:
        """Plot privacy-utility trade-off"""
        
        fig = go.Figure()
        
        # Utility curve
        fig.add_trace(go.Scatter(
            x=epsilon_values,
            y=utility_scores,
            mode='lines+markers',
            name='Model Utility',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        # Privacy curve
        fig.add_trace(go.Scatter(
            x=epsilon_values,
            y=privacy_scores,
            mode='lines+markers',
            name='Privacy Level',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        # Add optimal point annotation
        if epsilon_values and utility_scores and privacy_scores:
            # Find balance point (maximize utility * privacy)
            balance_scores = [u * p for u, p in zip(utility_scores, privacy_scores)]
            optimal_idx = balance_scores.index(max(balance_scores))
            
            fig.add_annotation(
                x=epsilon_values[optimal_idx],
                y=utility_scores[optimal_idx],
                text="Optimal Balance",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red")
            )
        
        fig.update_layout(
            title="Privacy-Utility Trade-off Analysis",
            xaxis_title="Epsilon (ε) - Privacy Parameter",
            yaxis_title="Score (0-1)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_byzantine_detection(self, detection_history: List[Dict[str, Any]]) -> go.Figure:
        """Plot Byzantine attack detection over time"""
        
        if not detection_history:
            return self._create_empty_plot("No Byzantine attack data available")
        
        df = pd.DataFrame(detection_history)
        
        fig = go.Figure()
        
        # Bar chart of attacks detected per round
        fig.add_trace(go.Bar(
            x=df.get('round', range(len(detection_history))),
            y=df.get('byzantine_detected', [0] * len(detection_history)),
            name='Byzantine Attacks Detected',
            marker_color='red',
            opacity=0.7
        ))
        
        # Add trend line
        if len(detection_history) > 1:
            z = np.polyfit(range(len(detection_history)), 
                          df.get('byzantine_detected', [0] * len(detection_history)), 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=df.get('round', range(len(detection_history))),
                y=p(range(len(detection_history))),
                mode='lines',
                name='Trend',
                line=dict(color='darkred', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Byzantine Attack Detection Over Time",
            xaxis_title="Training Round",
            yaxis_title="Attacks Detected",
            height=400
        )
        
        return fig
    
    def plot_communication_overhead(self, phases: List[str], data_sizes: List[float], 
                                  network_calls: List[int]) -> go.Figure:
        """Plot communication overhead by phase"""
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Data size bars
        fig.add_trace(
            go.Bar(
                x=phases,
                y=data_sizes,
                name='Data Size (KB)',
                marker_color='lightblue',
                opacity=0.8
            ),
            secondary_y=False
        )
        
        # Network calls line
        fig.add_trace(
            go.Scatter(
                x=phases,
                y=network_calls,
                mode='lines+markers',
                name='Network Calls',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="FL Phase", tickangle=45)
        fig.update_yaxes(title_text="Data Size (KB)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Network Calls", secondary_y=True)
        
        fig.update_layout(
            title="Communication Overhead Analysis",
            height=500
        )
        
        return fig
    
    def plot_participant_statistics(self, participant_stats: Dict[str, Any]) -> go.Figure:
        """Plot participant statistics and activity"""
        
        if not participant_stats:
            return self._create_empty_plot("No participant data available")
        
        # Create pie chart for participant types
        fig = go.Figure()
        
        # Example participant type distribution
        participant_types = ['Hospitals', 'Clinics', 'Research Centers', 'Emergency Centers']
        counts = [2, 3, 2, 1]  # Default distribution
        
        if 'participant_types' in participant_stats:
            participant_types = list(participant_stats['participant_types'].keys())
            counts = list(participant_stats['participant_types'].values())
        
        fig.add_trace(go.Pie(
            labels=participant_types,
            values=counts,
            hole=0.4,
            marker_colors=self.color_palette[:len(participant_types)]
        ))
        
        fig.update_layout(
            title="Participant Distribution by Type",
            height=400,
            annotations=[dict(text='Participants', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def plot_reputation_scores(self, reputation_data: Dict[str, float]) -> go.Figure:
        """Plot reputation scores for all participants"""
        
        if not reputation_data:
            return self._create_empty_plot("No reputation data available")
        
        participants = list(reputation_data.keys())
        scores = list(reputation_data.values())
        
        # Color code by reputation level
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green' for score in scores]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=participants,
            y=scores,
            marker_color=colors,
            name='Reputation Score',
            text=[f'{score:.3f}' for score in scores],
            textposition='auto'
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="Good Reputation Threshold")
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Minimum Reputation Threshold")
        
        fig.update_layout(
            title="Participant Reputation Scores",
            xaxis_title="Participant",
            yaxis_title="Reputation Score",
            height=500,
            yaxis=dict(range=[0, 1]),
            xaxis_tickangle=45
        )
        
        return fig
    
    def plot_system_health_dashboard(self, health_metrics: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive system health dashboard"""
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Security Status', 'Performance Metrics', 'Threat Level', 'Resource Usage'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "scatter"}]]
        )
        
        # Security status gauge
        security_score = health_metrics.get('security_score', 0.85)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=security_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Security Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if security_score > 0.8 else "orange"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Performance metrics
        performance_metrics = health_metrics.get('performance', {
            'Accuracy': 0.87,
            'Privacy': 0.92,
            'Efficiency': 0.78,
            'Robustness': 0.83
        })
        
        fig.add_trace(
            go.Bar(
                x=list(performance_metrics.keys()),
                y=list(performance_metrics.values()),
                marker_color=['green', 'blue', 'orange', 'purple']
            ),
            row=1, col=2
        )
        
        # Threat level gauge
        threat_level = health_metrics.get('threat_level', 0.15)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=threat_level * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Threat Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red" if threat_level > 0.5 else "yellow" if threat_level > 0.2 else "green"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Resource usage over time
        time_points = list(range(1, 25))  # 24 hours
        cpu_usage = [50 + 20 * np.sin(i/4) + np.random.normal(0, 5) for i in time_points]
        memory_usage = [60 + 15 * np.cos(i/3) + np.random.normal(0, 3) for i in time_points]
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=cpu_usage,
                mode='lines',
                name='CPU Usage',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=memory_usage,
                mode='lines',
                name='Memory Usage',
                line=dict(color='blue')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="System Health Dashboard",
            showlegend=False
        )
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color='gray'),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_real_time_metrics(self, current_metrics: Dict[str, Any]) -> go.Figure:
        """Create real-time metrics display"""
        
        # Create metrics cards layout
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=('Active Participants', 'Current Accuracy', 'Privacy Budget', 'Threat Status'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, 
                   {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Active participants
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=current_metrics.get('active_participants', 8),
                title={'text': "Active"},
                number={'font': {'size': 40, 'color': 'green'}}
            ),
            row=1, col=1
        )
        
        # Current accuracy
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=current_metrics.get('accuracy', 0.87) * 100,
                title={'text': "Accuracy %"},
                number={'font': {'size': 30}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'green'}}
            ),
            row=1, col=2
        )
        
        # Privacy budget remaining
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=current_metrics.get('privacy_budget_remaining', 0.75) * 100,
                title={'text': "Privacy Budget %"},
                number={'font': {'size': 30}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'blue'}}
            ),
            row=1, col=3
        )
        
        # Threat status
        threat_color = 'green' if current_metrics.get('threat_level', 'Low') == 'Low' else 'orange'
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=1 if current_metrics.get('threat_level', 'Low') == 'Low' else 0,
                title={'text': current_metrics.get('threat_level', 'Low')},
                number={'font': {'size': 20, 'color': threat_color}}
            ),
            row=1, col=4
        )
        
        fig.update_layout(
            height=300,
            title_text="Real-Time System Metrics"
        )
        
        return fig
