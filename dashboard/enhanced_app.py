import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from glob import glob

st.set_page_config(
    page_title="Enhanced LLM OCR Comparison Dashboard",
    page_icon="🔍",
    layout="wide"
)


def load_results():
    traditional_pattern = "../results/**/benchmark_results_*.json"
    enhanced_pattern = "../results/**/enhanced_benchmark_results_*.json"

    traditional_files = glob(traditional_pattern, recursive=True)
    enhanced_files = glob(enhanced_pattern, recursive=True)

    all_files = traditional_files + enhanced_files

    if not all_files:
        return None, None, None

    latest_file = max(all_files, key=os.path.getctime)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    workflow_type = "enhanced" if "enhanced" in latest_file or df.get('workflow_type', pd.Series()).iloc[
        0] == 'enhanced' else "traditional"

    return df, latest_file, workflow_type


def display_enhanced_metrics(df):
    st.markdown("###Schema Generation Metrics")

    if 'schema_generation_time' in df.columns:
        schema_times = df['schema_generation_time'].dropna()
        if not schema_times.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_schema_time = schema_times.mean()
                st.metric("Avg Schema Time", f"{avg_schema_time:.2f}s")

            with col2:
                min_schema_time = schema_times.min()
                st.metric("Min Schema Time", f"{min_schema_time:.2f}s")

            with col3:
                max_schema_time = schema_times.max()
                st.metric("Max Schema Time", f"{max_schema_time:.2f}s")

            fig_schema_time = px.histogram(
                df.dropna(subset=['schema_generation_time']),
                x='schema_generation_time',
                nbins=20,
                title="Schema Generation Time Distribution"
            )
            st.plotly_chart(fig_schema_time, use_container_width=True)

    if 'metadata' in df.columns:
        st.markdown("###Document Type Analysis")

        doc_types = []
        for metadata in df['metadata'].dropna():
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue

            if isinstance(metadata, dict):
                doc_type = metadata.get('document_type', 'unknown')
                doc_types.append(doc_type)

        if doc_types:
            doc_type_df = pd.DataFrame({'document_type': doc_types})
            doc_type_counts = doc_type_df['document_type'].value_counts()

            fig_doc_types = px.pie(
                values=doc_type_counts.values,
                names=doc_type_counts.index,
                title="Document Types Tested"
            )
            st.plotly_chart(fig_doc_types, use_container_width=True)


def display_schema_examples(df):
    st.markdown("###Generated Schema Examples")

    if 'generated_schema' in df.columns:
        schemas_with_data = df[df['generated_schema'].notna()]

        if not schemas_with_data.empty:
            sample_schemas = schemas_with_data.head(3)

            for idx, row in sample_schemas.iterrows():
                with st.expander(f"Schema for {os.path.basename(row['file_url'])}"):
                    try:
                        if isinstance(row['generated_schema'], str):
                            schema = json.loads(row['generated_schema'])
                        else:
                            schema = row['generated_schema']

                        st.json(schema)

                        if pd.notna(row['predicted_json']) and row['predicted_json']:
                            st.markdown("**Extracted Data:**")
                            if isinstance(row['predicted_json'], str):
                                try:
                                    extracted_data = json.loads(row['predicted_json'])
                                    st.json(extracted_data)
                                except:
                                    st.text(row['predicted_json'])
                            else:
                                st.json(row['predicted_json'])

                    except Exception as e:
                        st.error(f"Error displaying schema: {e}")


def main():
    st.title("Enhanced LLM OCR Comparison Dashboard")
    st.markdown("*Advanced workflow with dynamic schema generation*")
    st.markdown("---")

    data = load_results()
    if data[0] is None:
        st.error("No benchmark results found! Please run the benchmark first.")
        st.code("python test_enhanced_workflow.py")
        return

    df, results_file, workflow_type = data

    if workflow_type == "enhanced":
        st.success(f"Enhanced Workflow Results - Loaded from: {os.path.basename(results_file)}")
    else:
        st.info(f"Traditional Workflow Results - Loaded from: {os.path.basename(results_file)}")
        st.warning("This appears to be traditional workflow data. For enhanced features, run the enhanced benchmark.")

    st.sidebar.header("Filters")

    available_models = df['ocr_model'].unique()
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=available_models
    )

    if 'workflow_type' in df.columns:
        workflow_types = df['workflow_type'].unique()
        if len(workflow_types) > 1:
            selected_workflow = st.sidebar.selectbox(
                "Workflow Type",
                workflow_types,
                index=0
            )
            filtered_df = df[(df['ocr_model'].isin(selected_models)) &
                             (df['workflow_type'] == selected_workflow)]
        else:
            filtered_df = df[df['ocr_model'].isin(selected_models)]
    else:
        filtered_df = df[df['ocr_model'].isin(selected_models)]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_tests = len(filtered_df)
        st.metric("Total Tests", total_tests)

    with col2:
        successful_tests = len(filtered_df[filtered_df['error'].isna()])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    with col3:
        avg_json_accuracy = filtered_df['json_accuracy'].mean()
        if pd.notna(avg_json_accuracy):
            st.metric("Avg JSON Accuracy", f"{avg_json_accuracy:.3f}")
        else:
            st.metric("Avg JSON Accuracy", "N/A")

    with col4:
        if 'usage' in filtered_df.columns:
            total_times = []
            for usage in filtered_df['usage'].dropna():
                if isinstance(usage, str):
                    try:
                        usage = json.loads(usage)
                    except:
                        continue
                if isinstance(usage, dict) and 'duration' in usage:
                    total_times.append(usage['duration'])

            if total_times:
                avg_total_time = sum(total_times) / len(total_times)
                st.metric("Avg Total Time", f"{avg_total_time:.1f}s")
            else:
                st.metric("Avg Total Time", "N/A")

    if workflow_type == "enhanced":
        display_enhanced_metrics(filtered_df)

    st.markdown("##Performance Metrics")

    if 'json_accuracy' in filtered_df.columns and filtered_df['json_accuracy'].notna().any():
        fig_json = px.box(
            filtered_df.dropna(subset=['json_accuracy']),
            x='ocr_model',
            y='json_accuracy',
            title="JSON Accuracy by Model"
        )
        fig_json.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_json, use_container_width=True)

    if workflow_type == "enhanced":
        st.markdown("##Time Breakdown Analysis")

        time_data = []
        for _, row in filtered_df.dropna(subset=['usage']).iterrows():
            try:
                usage = row['usage']
                if isinstance(usage, str):
                    usage = json.loads(usage)

                schema_time = row.get('schema_generation_time', 0)
                total_time = usage.get('duration', 0)
                processing_time = total_time - schema_time if total_time > schema_time else total_time

                time_data.append({
                    'model': row['ocr_model'],
                    'schema_time': schema_time,
                    'processing_time': processing_time,
                    'total_time': total_time
                })
            except:
                continue

        if time_data:
            time_df = pd.DataFrame(time_data)

            fig_time = go.Figure()

            fig_time.add_trace(go.Bar(
                name='Schema Generation',
                x=time_df['model'],
                y=time_df['schema_time'],
                marker_color='lightblue'
            ))

            fig_time.add_trace(go.Bar(
                name='OCR + Extraction',
                x=time_df['model'],
                y=time_df['processing_time'],
                marker_color='darkblue'
            ))

            fig_time.update_layout(
                barmode='stack',
                title='Time Breakdown by Model',
                xaxis_title='Model',
                yaxis_title='Time (seconds)',
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig_time, use_container_width=True)

    if workflow_type == "enhanced":
        display_schema_examples(filtered_df)

    st.markdown("##Detailed Results")

    display_columns = [
        'file_url', 'ocr_model', 'extraction_model',
        'json_accuracy', 'error'
    ]

    if workflow_type == "enhanced":
        display_columns.extend(['schema_generation_time'])

    available_columns = [col for col in display_columns if col in filtered_df.columns]

    display_df = filtered_df[available_columns].copy()
    display_df['file_url'] = display_df['file_url'].apply(lambda x: os.path.basename(x) if x else "")

    if 'json_accuracy' in display_df.columns:
        display_df['json_accuracy'] = display_df['json_accuracy'].round(4)

    if 'schema_generation_time' in display_df.columns:
        display_df['schema_generation_time'] = display_df['schema_generation_time'].round(2)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.markdown("##Download Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Results CSV",
            data=csv,
            file_name=f"{workflow_type}_benchmark_results.csv",
            mime="text/csv"
        )

    with col2:
        if workflow_type == "enhanced":
            schemas_data = []
            for _, row in filtered_df.iterrows():
                if pd.notna(row.get('generated_schema')):
                    schemas_data.append({
                        'file_url': row['file_url'],
                        'ocr_model': row['ocr_model'],
                        'schema': row['generated_schema']
                    })

            if schemas_data:
                schemas_json = json.dumps(schemas_data, indent=2)
                st.download_button(
                    label="Download Generated Schemas",
                    data=schemas_json,
                    file_name="generated_schemas.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()