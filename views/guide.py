import streamlit as st

from components.theme import disclaimer, panel, section_header
from components.charts import (
    biomedical_visual_figure,
    conceptual_anatomy_figure,
    synthetic_xray_figure,
)
from config.constants import DEMO_FEATURE_NAMES


def _start_here():
    with panel("Start here"):
        st.markdown(
            """
            QureAI is an educational research prototype for tabular machine-learning
            experiments. Its checked-in example uses a synthetic Wisconsin-style
            diagnostic feature schema: 30 numeric measurements named after radius,
            texture, perimeter, area, smoothness, compactness, concavity, concave
            points, symmetry, and fractal dimension (mean, error, and worst groups).
            A `diagnosis` value may be `B` (benign) or `M` (malignant) in the example.
            These labels describe the dataset's categories only; they are not a
            clinically validated assessment, prevalence estimate, treatment decision,
            or diagnosis.
            """
        )
        st.info(
            "Recommended first run: open Data Workspace → load the built-in demo → "
            "confirm the binary diagnosis target → explore the 3D orientation → "
            "optionally run Experiment Lab → review Benchmark and Explainability."
        )


def _jury_walkthrough():
    with st.expander("Jury / first-time visitor: 5-minute walkthrough", expanded=True):
        st.markdown(
            """
            You do not need machine-learning or medical-data experience to try
            QureAI. Use the built-in synthetic dataset and follow this path:

            **1. Overview** — read the project purpose and the research-only
            disclaimer.

            **2. Data Workspace** — click **Load built-in demonstration dataset**.
            This loads synthetic, Wisconsin-style breast-cancer feature data; it is
            not a real patient record. Confirm that `diagnosis` is the selected
            target and that the validation message is healthy.

            **3. Explore the 3D orientation** — rotate the chart with your mouse.
            The points show mathematical similarity in the loaded measurements.
            It is not a 3D scan, an organ model, or a diagnosis.

            **4. Experiment Lab** — leave the default settings and click **Run
            Experiment**. The classical models should finish quickly. Leave the
            quantum option off for a fast demonstration; quantum simulation is an
            optional research comparison and can take longer.

            **5. Model Benchmark** — compare measured accuracy, F1, and ROC-AUC.
            These are results for this dataset and validation procedure only; the
            highest number does not prove clinical superiority.

            **6. Patient Analysis** — click **Load Example Patient**. Review the
            model outputs as an example of inference on one feature row, not as a
            real patient's diagnosis.

            **7. Explainability** — inspect which input measurements influenced the
            fitted model. This explains model behaviour, not biology or causation.
            """
        )
        disclaimer(
            "For a presentation, keep the synthetic-data label and research-only disclaimer visible."
        )


def render():
    section_header(
        "Guide & methodology",
        "A practical, non-diagnostic guide to the QureAI research console.",
        icon=":material/menu_book:",
    )

    _start_here()
    _jury_walkthrough()

    with st.expander("Interactive anatomical orientation (aesthetic only)", expanded=False):
        st.markdown(
            """
            This is a stylized 3D orientation to make the biomedical context
            approachable. It is not a scan, a patient avatar, an organ model,
            lesion localization tool, or a representation of the measurements.
            QureAI's actual models operate on tabular numeric features, not on
            this visual object.
            """
        )
        st.plotly_chart(
            conceptual_anatomy_figure(),
            width="stretch",
            config={"displaylogo": False},
        )
        disclaimer("Conceptual anatomy orientation only; it carries no patient-specific or diagnostic information.")

    with st.expander("Biomedical Visual Lab: tissue, tumor, and X-ray-style simulations", expanded=False):
        st.markdown(
            """
            These interactive views are generated procedurally inside the app to
            make biomedical concepts visually engaging. They are not real scans,
            pathology, patient anatomy, tumor segmentation, or model inputs.
            They should never be interpreted as showing what a benign or malignant
            lesion looks like in an individual.
            """
        )
        view = st.selectbox(
            "Visual simulation",
            ["Layered tissue", "Synthetic tumor model", "Synthetic X-ray projection"],
            key="biomedical_visual_view",
        )
        if view == "Synthetic X-ray projection":
            lesion_radius = st.slider(
                "Synthetic projection feature size",
                min_value=0.12,
                max_value=0.42,
                value=0.22,
                step=0.01,
                key="synthetic_xray_size",
            )
            st.plotly_chart(
                synthetic_xray_figure(lesion_radius),
                width="stretch",
                config={"displaylogo": False},
            )
        else:
            opacity = st.slider(
                "Tissue layer opacity",
                min_value=0.25,
                max_value=0.85,
                value=0.58,
                step=0.05,
                key="biomedical_visual_opacity",
            )
            lesion_radius = st.slider(
                "Synthetic nodule size",
                min_value=0.12,
                max_value=0.42,
                value=0.22,
                step=0.01,
                key="synthetic_nodule_size",
            )
            st.plotly_chart(
                biomedical_visual_figure(view, lesion_radius, opacity),
                width="stretch",
                config={"displaylogo": False},
            )
        disclaimer("Procedural educational simulation only; not a patient-specific image, diagnosis, or radiology result.")

    with st.expander("Medical researcher: what to inspect", expanded=False):
        st.markdown(
            """
            QureAI is most useful as a reproducible methods demonstration. Before
            interpreting a result, inspect:

            - **Dataset provenance:** whether the data are synthetic, retrospective,
              representative, and legally appropriate for the intended study.
            - **Schema and missingness:** whether measurements use compatible units,
              collection protocols, and missing-data assumptions.
            - **Validation design:** whether the split strategy prevents patient
              duplication and information leakage.
            - **Metrics together:** accuracy, precision, recall, F1, ROC-AUC,
              confusion matrices, class balance, and uncertainty—not one score.
            - **Generalization:** external validation, calibration, subgroup
              performance, distribution shift, and clinically meaningful thresholds.
            - **Explanations:** whether important features are plausible and stable;
              an attribution is not evidence of a biological mechanism.

            The current demonstration is suitable for learning and method
            comparison. It is not a validated clinical decision-support system and
            should not be used for screening, diagnosis, triage, or treatment.
            """
        )

    with st.expander("Beginner glossary: what do I need to know?", expanded=False):
        st.markdown(
            """
            - **Dataset:** a table of records. Each row is one sample; each column
              is a measurement or label.
            - **Feature:** an input measurement supplied to the model, such as
              radius or texture.
            - **Target:** the known outcome used during training, such as the demo
              labels `B` and `M`.
            - **Model:** a mathematical pattern-finding method trained on examples.
            - **Prediction:** the model's output for a row it did not use as a
              training label.
            - **Probability:** an estimator score for a class; it is not a
              patient's medical risk unless separately calibrated and validated.
            - **Confidence:** a descriptive model-strength indicator used by this
              prototype; it is not clinical certainty.
            - **Benchmark:** a side-by-side comparison of measured model results.
            - **Explainability:** a view of which inputs affected the fitted model's
              output; it is not a doctor's explanation of disease.
            - **Quantum option:** an experimental simulator-based comparison, not
              proof that quantum computing is better.
            """
        )

    with st.expander("How to use each page", expanded=True):
        st.markdown(
            """
            - **Overview** — read the workflow and research boundaries before starting.
            - **Data Workspace** — upload a CSV or load the demo, inspect validation,
              choose a target, and view the bounded feature-space orientation.
            - **Experiment Lab** — configure cross-validation and optionally run the
              slower quantum-kernel demonstration after validation passes.
            - **Quantum Analysis** — inspect a quantum circuit and kernel artifacts
              only when an opt-in run completed and Qiskit is available.
            - **Model benchmark** — compare measured classical and quantum metrics,
              stability, and runtime; do not treat the highest score as clinical proof.
            - **Patient Analysis** — compare one or more feature rows with the
              reference model. This is a research demonstration, not clinical care.
            - **Explainability** — inspect model feature importance or contributions;
              an explanation describes model behaviour, not biological causation.
            - **System Status** — check the Python runtime, demo CSV, persisted
              artifacts, and optional integrations.
            - **Limitations** — keep calibration, validation, distribution shift,
              simulator limits, and responsible-use boundaries in view.
            - **Guide & Methodology** — return here for definitions, input rules,
              examples, and troubleshooting.
            """
        )

    with st.expander("CSV input contract", expanded=False):
        st.markdown(
            """
            Use a comma-separated file with one record per row and a header row.
            Feature values must be numeric; ordinary blanks are missing values. The
            supervised training format is the 30 features below plus an optional
            target column (for example `diagnosis`). A patient prediction CSV uses
            the feature columns only and should not include a target.

            Headers are matched safely after trimming, lower-casing, and collapsing
            spaces/underscores. For example, `mean radius`, `Mean Radius`,
            `mean_radius`, and common UCI names such as `radius_mean` match.
            This normalization does not guess arbitrary synonyms, units, or
            misspellings.
            """
        )
        st.code(", ".join(DEMO_FEATURE_NAMES), language="text")
        st.markdown(
            """
            **Target behavior:** a target is optional for inspection, validation, and
            the orientation chart. Experiments and supervised predictions require a
            binary target in the reference data. Target classes are encoded for the
            experiment; the displayed class names remain dataset labels.

            **Columns:** reordering is accepted and aligned to the canonical feature
            order. Extra columns are ignored with a warning. Missing required
            features, duplicate/ambiguous headers, or non-numeric feature entries are
            errors. Blank numeric values can be median-imputed by preprocessing, but
            missingness should be reviewed rather than silently assumed harmless.
            """
        )

    with st.expander("Two safe demo flows", expanded=False):
        st.markdown(
            """
            **Dataset-to-benchmark flow**
            1. Load the built-in demonstration dataset in Data Workspace.
            2. Confirm `diagnosis` is selected and has two classes.
            3. Read the validation warnings and inspect the 3D orientation.
            4. In Experiment Lab, start with classical models and the default CV.
            5. Compare results in Model benchmark. Quantum execution is optional and
               bounded because kernel simulation can be expensive.

            **Patient-row smoke flow**
            1. Complete the dataset-to-benchmark setup or use the checked-in demo.
            2. Open Patient Analysis and choose **Load Example Patient**.
            3. Check the feature table and schema messages before reading the output.
            4. Use Explainability only to study model behaviour. Do not use the
               output to diagnose, triage, or recommend treatment.
            """
        )

    with panel("Methodology in plain language"):
        st.markdown(
            """
            1. **Validation** checks headers, duplicate names, binary targets, row
               counts, numeric values, and missing cells before a run.
            2. **Preprocessing** imputes numeric missing values using training-fold
               medians, applies `StandardScaler` so features are on comparable
               scales, and may apply PCA to create a smaller representation.
            3. **Classical models** are evaluated with stratified cross-validation.
               Each fold fits preprocessing only on its training rows to reduce
               leakage.
            4. **QSVC / quantum kernel** is an optional research demonstration. A
               quantum feature map produces a similarity (kernel) matrix, which a
               QSVC uses for classification. The bounded simulator workflow is not
               evidence of quantum advantage or hardware performance.
            5. **Reporting** shows metrics, runtime, confusion matrices, and
               diagnostic artifacts. A single dataset or split cannot establish
               generalization, calibration, clinical utility, or causality.
            """
        )

    with st.expander("Model terms", expanded=False):
        st.markdown(
            """
            - **Feature:** an input measurement used by a model, such as mean radius.
            - **Target:** the outcome column a supervised model learns to predict.
            - **Preprocessing:** repeatable preparation such as imputation, scaling,
              and dimensionality reduction before fitting a model.
            - **StandardScaler:** centers each feature and scales its training
              variation; it does not make measurements clinically comparable.
            - **PCA:** principal component analysis, a mathematical projection that
              compresses correlated features into components. Components are not
              anatomy.
            - **Probability:** a model score interpreted as a class probability by
              the estimator. It is not automatically calibrated risk.
            - **Model confidence:** informal shorthand for how strongly a model
              separates classes (for example, a probability away from 0.5). It is
              not certainty, clinical confidence, or a patient's chance of disease.
            - **ROC-AUC:** the area under the receiver-operating-characteristic
              curve, a ranking measure across thresholds. It is not prevalence,
              sensitivity at a chosen threshold, or clinical utility.
            - **Sensitivity:** the positive-class recall; how many positive labels
              were detected in the evaluation rows.
            - **Specificity:** the true-negative rate; how many negative labels
              were correctly rejected in the evaluation rows.
            - **QSVC / quantum kernel:** a support-vector classifier using similarities
              generated by a quantum feature map (or simulator).
            - **SHAP / contributions:** feature-attribution methods that distribute
              a model output across inputs. They explain the fitted model's
              behaviour, not biological cause.
            - **Leakage:** information from validation/test rows, future outcomes,
              duplicate records, or a target-derived feature entering training.
              Leakage can make scores look better than real-world performance.
            """
        )

    with st.expander("Troubleshooting", expanded=False):
        st.markdown(
            """
            - **Missing required features:** compare the header with the 30-name
              contract above; safe spacing/underscore and case variants are okay,
              but arbitrary abbreviations are not.
            - **Non-numeric values:** remove units, thousands separators, text
              placeholders, or convert them to blank values intentionally.
            - **Target is not binary:** select a target with exactly two non-null
              classes, or use the dataset for inspection only.
            - **No target found:** choose one explicitly in Data Workspace. Patient
              rows do not need a target, but their reference dataset does.
            - **Quantum unavailable/slow:** install the optional compatible
              dependencies, or continue with classical models; quantum simulation
              is intentionally opt-in and sample-bounded.
            - **Patient schema error:** upload feature columns from the reference
              schema, not the target, and check for duplicate headers.
            - **No persisted model:** Patient Analysis can train a compatible
              fallback in memory; this is slower and still not clinical validation.
            - **Blank chart or too few rows:** provide at least three usable numeric
              features and enough non-empty rows for three PCA coordinates.
            """
        )

    with panel("Research boundary"):
        st.markdown(
            """
            Use QureAI for learning, reproducible experimentation, and model
            comparison. The synthetic Wisconsin-style schema and its benign/malignant
            labels are educational categories, not a statement about a person's
            health. This prototype has no claim about prevalence, treatment,
            diagnosis, clinical validation, or clinical deployment. Do not upload
            identifiable health information and do not use any page to make care
            decisions. Consult qualified clinicians and validated, governed tools
            for real-world medical questions.
            """
        )
    disclaimer(
        "Non-diagnostic research prototype. Feature charts, probabilities, confidence-like scores, and explanations are not clinical evidence."
    )
