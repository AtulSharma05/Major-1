"""Unit tests for lightweight analytical result assembler."""

import pandas as pd

from src.agent.execution.engine import ExecutionResult
from src.agent.planning.assembler import ResultAssembler
from src.agent.planning.planner import AnalyticalPlan, PlanStep


def test_assemble_single_step_scalar():
    """Verify single-step scalar pass-through."""
    step = PlanStep(step_id=1, question="Total sales", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Total sales",
        steps=[step],
        execution_order=[1],
        is_multi_step=False,
    )
    res1 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore",
        dataframe=pd.DataFrame({"total_sales": [5000.0]}),
        scalar_value=5000.0,
        is_scalar=True,
        row_count=1,
        column_count=1,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1})

    assert assembled.all_steps_succeeded is True
    assert assembled.is_multi_step is False
    assert assembled.primary_scalar == 5000.0
    assert assembled.summary_metrics["result"] == 5000.0


def test_assemble_multi_step_scalar_comparison():
    """Verify combining multiple scalar steps into a comparison DataFrame with variance."""
    step1 = PlanStep(step_id=1, question="Sales in 2016", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Sales in 2017", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Compare 2016 vs 2017 sales",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    res1 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore WHERE date=2016",
        dataframe=pd.DataFrame({"s": [1000.0]}),
        scalar_value=1000.0,
        is_scalar=True,
        row_count=1,
        column_count=1,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore WHERE date=2017",
        dataframe=pd.DataFrame({"s": [1500.0]}),
        scalar_value=1500.0,
        is_scalar=True,
        row_count=1,
        column_count=1,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is True
    assert assembled.is_multi_step is True
    assert assembled.primary_dataframe is not None
    assert len(assembled.primary_dataframe) == 2
    assert assembled.summary_metrics["absolute_difference"] == 500.0
    assert assembled.summary_metrics["percentage_change"] == 50.0


def test_assemble_multi_step_tabular_merge():
    """Verify merging tabular steps sharing a common categorical key (e.g., 'region')."""
    step1 = PlanStep(
        step_id=1,
        question="Regional sales 2016",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Regional sales 2017",
        target_table="superstore",
        depends_on=[],
    )
    plan = AnalyticalPlan(
        original_question="Regional sales 2016 vs 2017",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    df1 = pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 200.0]})
    df2 = pd.DataFrame({"region": ["East", "West"], "sales": [150.0, 250.0]})

    res1 = ExecutionResult(
        success=True,
        query="SELECT region, sales...",
        dataframe=df1,
        is_scalar=False,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT region, sales...",
        dataframe=df2,
        is_scalar=False,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is True
    assert assembled.primary_dataframe is not None
    assert "region" in assembled.primary_dataframe.columns
    assert "sales_1" in assembled.primary_dataframe.columns
    assert "sales_2" in assembled.primary_dataframe.columns
    assert len(assembled.primary_dataframe) == 2


def test_assemble_multi_step_with_failure():
    """Verify handling when an intermediate step fails."""
    step1 = PlanStep(step_id=1, question="Step 1", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Step 2", target_table="superstore", depends_on=[1])
    plan = AnalyticalPlan(
        original_question="Multi-part query",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    res1 = ExecutionResult(success=True, query="SELECT 1", dataframe=pd.DataFrame({"a": [1]}))
    res2 = ExecutionResult(
        success=False,
        query="BAD SQL",
        error_type="SyntaxError",
        error_message="Fail",
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is False
    assert any("failed" in note.lower() for note in assembled.assembly_notes)


def test_assemble_scalar_comparison_exact_variance_and_pct_change():
    """Verify scalar variance and percentage change calculation using 2016 vs 2017 figures."""
    step1 = PlanStep(
        step_id=1,
        question="Total sales 2016",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Total sales 2017",
        target_table="superstore",
        depends_on=[],
    )
    plan = AnalyticalPlan(
        original_question="Compare 2016 and 2017 total sales",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    res1 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore WHERE order_date ... 2016",
        dataframe=pd.DataFrame({"sales_2016": [2615.97]}),
        scalar_value=2615.97,
        is_scalar=True,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore WHERE order_date ... 2017",
        dataframe=pd.DataFrame({"sales_2017": [2162.41]}),
        scalar_value=2162.41,
        is_scalar=True,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is True
    assert assembled.summary_metrics["step_1_value"] == 2615.97
    assert assembled.summary_metrics["step_2_value"] == 2162.41
    # 2017 - 2016 = 2162.41 - 2615.97 = -453.56
    assert assembled.summary_metrics["absolute_difference"] == -453.56
    # (-453.56 / 2615.97) * 100 = -17.338... -> -17.34%
    assert assembled.summary_metrics["percentage_change"] == -17.34
    assert len(assembled.assembly_notes) > 0


def test_assemble_scalar_comparison_zero_baseline():
    """Verify safe handling of zero baseline without ZeroDivisionError."""
    step1 = PlanStep(step_id=1, question="Baseline", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Current", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Compare baseline vs current",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    res1 = ExecutionResult(
        success=True,
        query="SELECT 0.0",
        dataframe=pd.DataFrame({"v": [0.0]}),
        scalar_value=0.0,
        is_scalar=True,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT 500.0",
        dataframe=pd.DataFrame({"v": [500.0]}),
        scalar_value=500.0,
        is_scalar=True,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is True
    assert assembled.summary_metrics["step_1_value"] == 0.0
    assert assembled.summary_metrics["step_2_value"] == 500.0
    assert assembled.summary_metrics["absolute_difference"] == 500.0
    assert assembled.summary_metrics["percentage_change"] is None


def test_assembled_result_dashboard_contract():
    """Verify that AssembledResult fulfills all attributes accessed by the dashboard."""
    step = PlanStep(step_id=1, question="Q", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Q",
        steps=[step],
        execution_order=[1],
        is_multi_step=False,
    )
    res = ExecutionResult(
        success=True,
        query="SELECT 1",
        dataframe=pd.DataFrame({"a": [1]}),
        scalar_value=1,
        is_scalar=True,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res})

    # Dashboard contract checks:
    assert hasattr(assembled, "assembly_notes")
    assert isinstance(assembled.assembly_notes, list)
    assert hasattr(assembled, "all_steps_succeeded")
    assert hasattr(assembled, "summary_metrics")
    assert hasattr(assembled, "primary_dataframe")
    assert hasattr(assembled, "primary_scalar")


def test_assemble_independent_rankings_retains_both_answers():
    """Verify that independent ranking steps retain both distinct answers in assembled output."""
    step1 = PlanStep(
        step_id=1,
        question="Find sub_category with highest sales",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Find sub_category with highest profit",
        target_table="superstore",
        depends_on=[],
    )
    plan = AnalyticalPlan(
        original_question="Which product has highest sales and which has highest profit?",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    df_sales = pd.DataFrame({"sub_category": ["Chairs"], "total_sales": [731.94]})
    df_profit = pd.DataFrame({"sub_category": ["Phones"], "total_profit": [210.71]})

    res1 = ExecutionResult(
        success=True,
        query="SELECT sub_category, SUM(sales)...",
        dataframe=df_sales,
        is_scalar=False,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT sub_category, SUM(profit)...",
        dataframe=df_profit,
        is_scalar=False,
    )

    assembler = ResultAssembler()
    assembled = assembler.assemble(plan, {1: res1, 2: res2})

    assert assembled.all_steps_succeeded is True
    assert assembled.is_multi_step is True

    # Check that both answers are retained in summary_metrics
    assert assembled.summary_metrics["step_1_winner"] == "Chairs"
    assert assembled.summary_metrics["step_2_winner"] == "Phones"

    # Check that both distinct entities are present in primary_dataframe
    assert assembled.primary_dataframe is not None
    entities = list(assembled.primary_dataframe["sub_category"])
    assert "Chairs" in entities
    assert "Phones" in entities
    assert len(assembled.primary_dataframe) == 2

    # Verify granularity disclosure note exists
    assert any("sub_category" in n for n in assembled.assembly_notes)
