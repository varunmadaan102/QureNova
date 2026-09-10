from components.charts import biomedical_visual_figure, synthetic_xray_figure


def test_biomedical_visuals_are_generated_without_external_assets():
    tissue = biomedical_visual_figure("Layered tissue")
    tumor = biomedical_visual_figure("Synthetic tumor model", lesion_radius=0.3)
    projection = synthetic_xray_figure(lesion_radius=0.3)

    assert len(tissue.data) == 3
    assert len(tumor.data) == 3
    assert len(projection.data) == 1
