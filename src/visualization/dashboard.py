"""
Streamlit Dashboard
==================

Interactive training visualization dashboard.

Usage:
    streamlit run src/visualization/dashboard.py

Author: Combat Racing RL Team
Date: 2024-2025
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Combat Racing RL Dashboard",
    page_icon="🏁",
    layout="wide"
)

# Title
st.title("🏁 Combat Racing Championship - Training Dashboard")
st.markdown("Real-time monitoring and analysis of RL agent training")

# Sidebar - Agent selection
st.sidebar.header("📊 Configuration")

# Find available agents
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    agents = [d.name for d in checkpoint_dir.iterdir() if d.is_dir()]
else:
    agents = []

selected_agent = st.sidebar.selectbox(
    "Select Agent",
    agents if agents else ["No agents found"],
    help="Choose agent to visualize"
)

# Load metrics if agent selected
if agents and selected_agent != "No agents found":
    metrics_path = checkpoint_dir / selected_agent / "metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        episode_rewards = metrics.get('episode_rewards', [])
        episode_lengths = metrics.get('episode_lengths', [])
        eval_rewards = metrics.get('eval_rewards', [])
        episode_times = metrics.get('episode_times', [])
        
        # Metrics summary
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Summary Stats")
        
        if episode_rewards:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Episodes", len(episode_rewards))
                st.metric("Mean Reward", f"{np.mean(episode_rewards[-100:]):.2f}")
            with col2:
                st.metric("Max Reward", f"{np.max(episode_rewards):.2f}")
                st.metric("Best Eval", f"{np.max(eval_rewards):.2f}" if eval_rewards else "N/A")
        
        # Main content - Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Training Progress",
            "📈 Detailed Metrics",
            "🔍 Analysis",
            "⚙️ Configuration"
        ])
        
        # Tab 1: Training Progress
        with tab1:
            st.header("Training Progress")
            
            # Episode rewards
            st.subheader("Episode Rewards")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Episode Rewards", "Episode Lengths",
                              "Evaluation Performance", "Reward Distribution")
            )
            
            # Plot 1: Episode rewards with moving average
            episodes = np.arange(len(episode_rewards))
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=episode_rewards,
                    mode='lines',
                    name='Episode Reward',
                    line=dict(color='lightblue', width=1),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            # Moving average
            window = 50
            if len(episode_rewards) > window:
                moving_avg = np.convolve(
                    episode_rewards,
                    np.ones(window) / window,
                    mode='valid'
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(window - 1, len(episode_rewards)),
                        y=moving_avg,
                        mode='lines',
                        name=f'Moving Avg ({window})',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Episode lengths
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=episode_lengths,
                    mode='lines',
                    name='Episode Length',
                    line=dict(color='green', width=1),
                    opacity=0.5
                ),
                row=1, col=2
            )
            
            # Plot 3: Evaluation rewards
            if eval_rewards:
                eval_episodes = np.linspace(0, len(episode_rewards), len(eval_rewards))
                fig.add_trace(
                    go.Scatter(
                        x=eval_episodes,
                        y=eval_rewards,
                        mode='lines+markers',
                        name='Evaluation',
                        line=dict(color='purple', width=2),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Reward distribution
            fig.add_trace(
                go.Histogram(
                    x=episode_rewards,
                    nbinsx=50,
                    name='Distribution',
                    marker=dict(color='orange')
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            fig.update_xaxes(title_text="Episode", row=1, col=1)
            fig.update_xaxes(title_text="Episode", row=1, col=2)
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_xaxes(title_text="Reward", row=2, col=2)
            fig.update_yaxes(title_text="Reward", row=1, col=1)
            fig.update_yaxes(title_text="Steps", row=1, col=2)
            fig.update_yaxes(title_text="Reward", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Detailed Metrics
        with tab2:
            st.header("Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reward Statistics")
                
                # Create dataframe
                df = pd.DataFrame({
                    'Episode': episodes,
                    'Reward': episode_rewards,
                    'Length': episode_lengths,
                })
                
                # Add rolling statistics
                df['Reward_MA50'] = df['Reward'].rolling(window=50).mean()
                df['Reward_MA100'] = df['Reward'].rolling(window=100).mean()
                
                st.dataframe(df.tail(20), use_container_width=True)
            
            with col2:
                st.subheader("Performance Trends")
                
                # Best/worst episodes
                if episode_rewards:
                    best_idx = np.argmax(episode_rewards)
                    worst_idx = np.argmin(episode_rewards)
                    
                    st.metric("Best Episode", f"#{best_idx}", 
                             f"{episode_rewards[best_idx]:.2f} reward")
                    st.metric("Worst Episode", f"#{worst_idx}",
                             f"{episode_rewards[worst_idx]:.2f} reward")
                    
                    # Recent performance
                    recent_mean = np.mean(episode_rewards[-100:])
                    early_mean = np.mean(episode_rewards[:100]) if len(episode_rewards) > 100 else 0
                    improvement = recent_mean - early_mean
                    
                    st.metric("Improvement", f"{improvement:.2f}",
                             f"Last 100 vs First 100")
        
        # Tab 3: Analysis
        with tab3:
            st.header("Performance Analysis")
            
            # Convergence analysis
            st.subheader("Convergence Analysis")
            
            if len(episode_rewards) > 200:
                # Split into segments
                segments = 4
                segment_size = len(episode_rewards) // segments
                segment_means = [
                    np.mean(episode_rewards[i*segment_size:(i+1)*segment_size])
                    for i in range(segments)
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Segment {i+1}" for i in range(segments)],
                    y=segment_means,
                    marker=dict(color=segment_means, colorscale='Viridis')
                ))
                fig.update_layout(
                    title="Mean Reward by Training Segment",
                    xaxis_title="Training Segment",
                    yaxis_title="Mean Reward",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Learning curve
            st.subheader("Learning Curve")
            
            fig = go.Figure()
            
            # Plot with confidence bands
            window = 100
            if len(episode_rewards) > window:
                moving_avg = np.convolve(
                    episode_rewards,
                    np.ones(window) / window,
                    mode='valid'
                )
                moving_std = pd.Series(episode_rewards).rolling(window).std()[window-1:]
                
                x = np.arange(window - 1, len(episode_rewards))
                
                # Confidence band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([
                        moving_avg + moving_std,
                        (moving_avg - moving_std)[::-1]
                    ]),
                    fill='toself',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Band'
                ))
                
                # Mean
                fig.add_trace(go.Scatter(
                    x=x,
                    y=moving_avg,
                    mode='lines',
                    name='Mean',
                    line=dict(color='blue', width=3)
                ))
            
            fig.update_layout(
                title="Learning Curve with Confidence Band",
                xaxis_title="Episode",
                yaxis_title="Reward",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Configuration
        with tab4:
            st.header("Agent Configuration")
            
            # Show agent info
            st.subheader("Training Info")
            
            info_data = {
                "Agent Type": selected_agent.upper(),
                "Total Episodes": len(episode_rewards),
                "Total Steps": sum(episode_lengths) if episode_lengths else 0,
                "Training Time": f"{sum(episode_times)/3600:.2f} hours" if episode_times else "N/A",
            }
            
            for key, value in info_data.items():
                st.text(f"{key}: {value}")
            
            # Download data
            st.subheader("Export Data")
            
            if st.button("Download Metrics JSON"):
                st.download_button(
                    label="Download",
                    data=json.dumps(metrics, indent=2),
                    file_name=f"{selected_agent}_metrics.json",
                    mime="application/json"
                )
    
    else:
        st.warning(f"No metrics found for agent: {selected_agent}")
        st.info("Train an agent first to see metrics here.")

else:
    st.info("👈 Train agents to see their performance here!")
    
    st.markdown("""
    ### Getting Started
    
    1. Train an agent:
    ```bash
    python -m src.training.train --agent dqn --episodes 1000
    ```
    
    2. Metrics will appear in `checkpoints/<agent>/metrics.json`
    
    3. Refresh this dashboard to see results!
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Combat Racing Championship**")
st.sidebar.markdown("Built for ENSAM Morocco 🇲🇦")
