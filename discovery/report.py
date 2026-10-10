from pathlib import Path
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image

from .common import read_json, read_jsonl
from .experiments import load_arrays


def figures(root):
    output = root / "results" / "figures"
    output.mkdir(exist_ok=True)
    values = load_arrays(root / "results" / "comparison.npz")
    x, t = values["x"], values["t"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
    for ax, name, title in zip(axes, ("truth", "discovered", "neural"), ("Reference field", "Discovered-law PINN", "Data-only neural field")):
        p = ax.pcolormesh(x, t, values[name], shading="auto", vmin=values["truth"].min(), vmax=values["truth"].max(), cmap="viridis")
        ax.axhline(0.8, color="white", linestyle="--", linewidth=1)
        ax.set(xlabel="Position", ylabel="Time", title=title)
    fig.colorbar(p, ax=axes, label="Concentration", shrink=0.85)
    fig.savefig(output / "fields.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    for name, label in (("discovered", "Discovery PINN"), ("neural", "Neural baseline"), ("oracle", "Oracle PINN")):
        error = np.linalg.norm(values[name] - values["truth"], axis=1) / np.linalg.norm(values["truth"], axis=1)
        axes[0].plot(t, 100 * error, label=label)
    axes[0].axvline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Time", ylabel="Relative field error (%)", title="Frozen-model extrapolation")
    axes[0].legend(fontsize=7)
    for i, time in ((0, 0), (60, 0.6), (120, 1.2)):
        line = axes[1].plot(x, values["forward_truth"][i], label=f"Reference t={time:g}")[0]
        axes[1].plot(x, values["forward"][i], "--", color=line.get_color(), linewidth=1.2)
    axes[1].set(xlabel="Position", ylabel="Concentration", title="New initial condition: forward PINN")
    axes[1].legend(fontsize=7)
    fig.savefig(output / "prediction.png", dpi=180)
    plt.close(fig)
    study = read_json(root / "results" / "sparse_study.json")["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    for sensors, marker in ((8, "o"), (16, "s")):
        subset = [r for r in study if r["sensors"] == sensors]
        noise = [100 * r["noise_fraction"] for r in subset]
        axes[0].plot(noise, [r["support_recovery_fraction"] for r in subset], marker=marker, label=f"{sensors} sensors")
        axes[1].plot(noise, [r["mean_parameter_error"] for r in subset], marker=marker, label=f"{sensors} sensors")
    axes[0].set(xlabel="Noise / signal RMS (%)", ylabel="Exact-support recovery fraction", ylim=(-0.05, 1.05))
    axes[1].set(xlabel="Noise / signal RMS (%)", ylabel="Joint relative coefficient error")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "sparse.png", dpi=180)
    plt.close(fig)
    return output


def make_report(root):
    root = Path(root)
    metrics = read_json(root / "results" / "metrics.json")
    benchmark = read_json(root / "results" / "benchmark.json")["summary"]
    chain = read_json(root / "results" / "ablation_chain.json")
    candidates = read_json(root / "results" / "candidates.json")
    equation = read_json(root / "results" / "equation.json")
    search = read_json(root / "results" / "search.json")
    dataset = read_json(root / "data" / "dataset.json")
    figure_dir = figures(root)
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (("Report", "DejaVuSans.ttf"), ("ReportBold", "DejaVuSans-Bold.ttf"),
                           ("ReportMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold",
                                 italic="Report", boldItalic="ReportBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", fontName="Report", fontSize=10.4, leading=14.2, spaceAfter=9))
    styles.add(ParagraphStyle(name="Heading", fontName="ReportBold", fontSize=15, leading=19, spaceAfter=13, textColor=colors.HexColor("#173450")))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["Body"], fontSize=8.4, leading=11.5, spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle(name="Equation", fontName="ReportMono", fontSize=8.2, leading=12, spaceAfter=11, backColor=colors.HexColor("#edf2f5"), borderPadding=8))
    story = []

    def title(text):
        story.append(Paragraph(escape(text), styles["Heading"]))

    def paragraph(text, small=False):
        story.append(Paragraph(text, styles["SmallBody" if small else "Body"]))

    def code(text):
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), styles["Equation"]))

    def picture(name, caption):
        story.append(Image(str(figure_dir / name), width=17 * cm, height=5.8 * cm))
        paragraph(caption, small=True)

    inverse, forward, uq = metrics["inverse"], metrics["forward_new_initial_condition"], metrics["uncertainty"]
    title("Generative Scientific Discovery Engine")
    paragraph("<b>Research prototype and executed evaluation</b><br/>Scalar transport and reaction dynamics from sparse noisy observations")
    paragraph("This prototype investigates an end-to-end route from scientific evidence and measurements to candidate laws, physics-informed parameter estimation, rejection, and prediction. Its contribution is the integrated, auditable discovery workflow. The controlled laws used for evaluation are established model families; identifying a law withheld from the inference engine is not a claim of discovering previously unknown science.")
    paragraph(f"The main run generated three distinct hypotheses and evaluated {search['evaluated_supports']} evidence-seeded term supports. It selected the correct four-term support without receiving the true coefficients or exact initial profile. The selected inverse PINN achieved {100 * inverse['prediction_relative_error']:.3f}% relative field error on the observation interval, {100 * inverse['ood_relative_error']:.3f}% on future times, and {100 * inverse['parameter_relative_error']:.3f}% joint coefficient error. A new-initial-condition forward solve achieved {100 * forward['prediction_relative_error']:.3f}% error without interior solution labels.")
    paragraph("The research artifact includes generated datasets, cited evidence, a provenance graph, raw pretrained-model responses, discovered equations, trained checkpoints, numerical predictions, an uncertainty study, and a 95-item benchmark. The benchmark separates 50 scientific questions, 25 evidence-combination questions, and 20 adversarial physics cases. Failed low-information and high-noise experiments remain in the results.")
    paragraph("The supported domain is a scalar field in a one-dimensional periodic channel, together with a spatially uniform ODE case. The grammar is bounded to first time derivatives, first and second spatial derivatives, polynomial reaction up to degree three, concentration-dependent transport, and a constant source. This explicit scope makes the discovery claim testable and avoids unsupported generalization to arbitrary physics.")
    paragraph("The implementation uses a local language model and sentence encoder for evidence discovery, NumPy/SciPy for weak identification and numerical integration, and PyTorch for trainable PINNs. The saved default run can replay exact pretrained requests offline while recomputing the physics and evaluation.")
    story.append(PageBreak())

    title("1. Evidence, graph and generative search")
    paragraph("The corpus contains 28 original evidence paraphrases from eleven scientific sources. MIT transport lectures provide the transport, reaction and boundary context [1-4]. Logistic dynamics supply a nonlinear reaction prior [5]. Sparse identification, weak formulations, and PINN studies guide the computational methods [6-11]. Every record retains a source URL, location, assumptions, and stable evidence ID.")
    paragraph("Retrieval combines normalized MiniLM embeddings with lexical overlap. The knowledge graph links evidence to physical concepts. Traversal expands the retrieved context through shared concepts, preserving the evidence-concept-evidence paths. Both retrieval benchmark arms use the same final chunk budget, so graph augmentation changes the selected evidence rather than simply supplying an unlimited context.")
    paragraph("Qwen2.5-1.5B-Instruct proposes equation structures. The model's citations and explanatory prose are not trusted: the graph resolves physical support independently, and the term semantics define the explanation. The raw model output is retained, including errors that the small model makes in explaining operators. A structured parse and bounded retries prevent malformed proposals from entering the scientific pipeline.")
    paragraph("The three supported initial proposals combine diffusion with drift, drift with linear reaction, and drift with quadratic reaction. None supplies the true four-term law. Two rounds of term addition and removal generate additional structures. Canonical support ordering prevents duplicates; the equation representation never executes arbitrary strings.")
    code("Allowed terms: u_xx, u_x, u, u2, u3, u*u_x, 1\nMaximum active terms: 4\nCoefficient dimensions = dimensions(u_t) - dimensions(term)")
    paragraph("Units are represented as exponents of concentration, length and time. Diffusivity has dimension L squared per T; the quadratic coefficient has dimension inverse concentration per T. Dimensional consistency and forward diffusivity admissibility are hard checks. Because an unknown dimensional coefficient can make many terms dimensionally valid, units alone do not identify the law.")
    paragraph("The engine first screens structures using measurements, then trains shortlisted inverse PINNs. Data fit, residual, integrated balance, complexity and bootstrap regression sensitivity contribute to final ranking. Initial/boundary compatibility and dimensional admissibility are enforced by the architecture and symbolic checks. A low PDE residual by itself cannot certify a candidate.")
    story.append(PageBreak())

    title("2. Numerical and neural methods")
    paragraph("The observation-only fit combines a spatial Fourier basis with temporal Chebyshev polynomials and ridge regularization. The spatial order is limited by the available training-sensor rank. The weak regression then integrates the reconstructed field against compact polynomial test functions. Integration by parts moves derivatives onto the test functions, reducing dependence on direct differentiation of noisy observations [7]. Reconstruction error remains a source of uncertainty.")
    code("-integral(u * phi_t) =\n D*integral(u * phi_xx) - c_x*integral(u * phi_x)\n + c_1*integral(u * phi) + c_2*integral(u2 * phi) + ...")
    paragraph("Seven temporal patch centres and nine spatial centres form 63 weak rows. Support coefficients are estimated with scaled constrained least squares. Blocked weak validation and finite-difference trajectory checks screen the candidates. The latter use reconstructed initial values and withheld sensor measurements, not a reference solution. Complexity weights scale with the assumed observation noise and selection-set size.")
    paragraph("The PINN has two trainable 32-unit tanh layers mapping normalized time to spatial Fourier amplitudes. Its Fourier representation enforces periodic field values and gradients. Autograd computes time derivatives; exact Fourier derivatives compute the first and second spatial derivatives. Independent tests compare these derivatives with direct coordinate autograd. The inverse model jointly trains field weights and physical coefficients; diffusivity uses a softplus parameterization.")
    code("L = 50*L_data + 50*L_initial + L_physics\nL_physics = mean(((u_t - equation_rhs)*T_scale/U_scale)**2)")
    paragraph("The data term is absent in forward mode. The exact initial profile is absent in inverse mode, where the learned field at time zero estimates it. Adam first fits observations and then activates physics, followed by L-BFGS refinement. The conventional neural baseline has the same field architecture and observation split with the physical loss removed.")
    paragraph("Physics validation samples new collocation points independently of training. For periodic fields the integrated flux cancels, so the balance compares total accumulation with integrated reaction. It does not incorrectly impose constant mass on a reacting field. Reference PDE data use pseudospectral derivatives, while validation rollouts use a separate second-order finite-difference operator; grid refinement and an analytical heat solution test their accuracy.")
    story.append(PageBreak())

    title("3. Main discovery and inverse results")
    picture("fields.png", "The dashed line marks the end of observations and training collocation. Values beyond it are predictions from frozen field networks.")
    paragraph(f"The main observations contain {dataset['training_points']} training measurements, {dataset['selection_points']} selection measurements, and {dataset['calibration_points']} calibration measurements. These are sampled from a dense 128-by-121 reference trajectory with independent Gaussian noise at 1% of signal RMS. Every fourth sensor is withheld; its earlier times select models and its later times calibrate prediction intervals.")
    code(equation["expression"])
    paragraph("The evaluator's hidden coefficients are 0.012, -0.16, 0.9 and -0.9 for diffusion, drift, linear reaction and quadratic reaction respectively. They remain under the evaluation data directory and never enter the inference input. The inverse initial profile is also inferred rather than supplied.")
    paragraph(f"Independent PDE-residual MSE is {inverse['pde_residual_mse']:.6g}; training-observation MSE is {inverse['data_loss']:.6g}; withheld-observation MSE is {inverse['validation_data_loss']:.6g}. Periodic boundary loss is {inverse['boundary_loss']:.3g}, integrated-balance RMS is {inverse['balance_rms']:.6g}, and inferred-initial-profile MSE is {inverse['initial_loss']:.6g}. Relative errors for the individual coefficients are " + ", ".join(f"{name}: {100 * error:.3f}%" for name, error in inverse["coefficient_errors"].items()) + ".")
    rejected = [c for c in candidates["pinn_trials"] if c["status"] == "rejected"]
    paragraph(f"The diffusion-only challenger is rejected because its withheld-observation discrepancy exceeds three times the assumed noise variance. Its measured validation MSE is {rejected[0]['metrics']['validation_data_loss']:.6g}. Several richer alternatives remain competitive; selection of the correct support in this controlled run is not proof that the sparse data uniquely determine it.")
    story.append(PageBreak())

    title("4. Prediction, ODE discovery and uncertainty")
    picture("prediction.png", "Left: each frozen field is evaluated beyond the observation window. Right: dashed curves are the physics-only forward solve for a changed initial condition.")
    paragraph("The temporal OOD test evaluates the frozen inverse model from time 0.8 to 1.2, where neither labels nor collocation points were supplied during training. A separate forward problem changes the initial profile, freezes discovered coefficients, and trains a new field against physics and initial constraints over the prediction domain. These are distinct generalization tests. Prediction arrays are committed by hash before the corresponding references are loaded for scoring.")
    ode = metrics["ode"]
    paragraph(f"The time-only experiment identifies {escape(ode['equation']['expression'])}. Its joint coefficient error is {100 * ode['parameter_relative_error']:.3f}% and its in-domain field error is {100 * ode['prediction_relative_error']:.3f}%. The evaluator uses an analytical logistic reference. This experiment exercises ODE identification and the same inverse-PINN parameter machinery without spatial terms.")
    paragraph(f"A three-member fitted ensemble, including resampled observations, reports coefficient dispersion and field uncertainty. A multiplier of {uq['calibration_multiplier']:.3f} is calibrated on {uq['calibration_size']} reserved noisy measurements for nominal {100 * uq['nominal_coverage']:.0f}% predictive coverage. The measured clean-reference coverage is {100 * uq['clean_reference_coverage_id']:.1f}% in-domain and {100 * uq['clean_reference_coverage_ood']:.1f}% out-of-domain. Mean interval width is {uq['mean_interval_width']:.5f}. These are empirical results for this experiment, not guarantees for noisy future observations or distribution shifts.")
    structural = metrics["structure_uncertainty"]
    paragraph(f"Twelve observation bootstrap runs estimate structural stability. The finally selected support appears in {100 * structural['selected_support_frequency']:.1f}% of weak-identification selections. This result exposes ambiguity between similarly fitting reaction models. Frequencies describe this resampling procedure, not posterior model probabilities; coefficient uncertainty is conditional on the selected support.")
    story.append(PageBreak())

    title("5. Sparse/noisy experiments and ablations")
    picture("sparse.png", "All eighteen weak-identification experiments are included, spanning two sensor densities, three noise levels and three seeds. One seed per regime also receives PINN refinement.")
    paragraph("Support recovery degrades in the lower-density and higher-noise regimes. The failure results are preserved rather than discarded. In the 16-sensor cases the grammar identifies the true support across the tested seeds at zero and 1% noise, while 5% noise introduces competing structures. At eight sensors, limited spatial observability and reconstruction bias prevent consistently reliable identification.")
    paragraph("The required four-stage ablation uses the same main observations and reference. RAG-only and RAG+KG supply equation structures; a common weak-coefficient fitter and numerical solver evaluate their proposals. The predefined PINN fixes the best graph proposal's support and infers its coefficients. The full engine adds generative support search and PINN validation. Thus coefficient fitting used to score a language proposal is not misrepresented as a language-model prediction.")
    for name, label in (("llm_rag", "LLM/RAG"), ("rag_kg", "RAG+KG"), ("predefined_pinn", "Fixed-support PINN"), ("generative_pinn_discovery", "Generative PINN discovery")):
        r = chain[name]
        paragraph(f"<b>{label}:</b> field error {100 * r['prediction_relative_error']:.3f}%, temporal OOD error {100 * r['ood_relative_error']:.3f}%, coefficient error {100 * r['parameter_relative_error']:.3f}%; exact support recovery: {r['support_exact']}.", small=True)
    a = metrics["ablations"]
    paragraph(f"Without PINN refinement, joint coefficient error is {100 * a['without_pinn_refinement']['parameter_relative_error']:.3f}%, versus {100 * inverse['parameter_relative_error']:.3f}% for the full method. The data-only neural baseline has {100 * a['conventional_neural_network']['ood_relative_error']:.3f}% future error. The oracle PINN receives the true support and is an explicit upper-reference comparator. Removing evidence seeding expands support evaluation from {search['evaluated_supports']} to {a['without_evidence_seeding']['evaluated_supports']}; the effect is search efficiency rather than demonstrated superiority on every accuracy metric.")
    story.append(PageBreak())

    title("6. Benchmark, checks and practical scope")
    for category, label in (("scientific", "Scientific questions"), ("multi_hop", "Multi-hop questions")):
        r = benchmark[category]
        paragraph(f"<b>{label}, n={r['count']}:</b> RAG answer accuracy {100 * r['rag']['answer_accuracy']:.1f}%, RAG+KG accuracy {100 * r['rag_kg']['answer_accuracy']:.1f}%; concept coverage {100 * r['rag']['concept_fraction']:.1f}% and {100 * r['rag_kg']['concept_fraction']:.1f}%; evidence recall {100 * r['rag']['evidence_recall']:.1f}% and {100 * r['rag_kg']['evidence_recall']:.1f}% respectively.")
    paragraph(f"The adversarial validator passes {100 * benchmark['adversarial']['accuracy']:.1f}% of twenty cases. These exercise unit contradictions, negative forward diffusivity, invalid expressions, incorrect mass balances, signed quantities, incompatible conditions, rank-deficient inverse problems, heterogeneous noise, fabricated citations, contradictory assumptions, unresolved spatial modes, and unsupported OOD certainty. Expected outcomes include rejection, insufficient information, qualification and acceptance; the validator does not simply reject every challenge.")
    paragraph("Text scoring uses deterministic synonym groups and resolved evidence citations. It is an auditable heuristic rather than expert semantic grading. The raw responses, contexts and paths allow independent inspection. Missing citations and unsupported answers count against the baseline. Scientific prose from the small model is not accepted as the engine's validation authority.")
    checks = metrics["numerical_checks"]
    paragraph(f"The analytical heat-equation comparison gives relative error {checks['heat_exact_relative_error']:.6g}. The finite-difference reference errors at 64, 128 and 256 grid points are " + ", ".join(f"{r['relative_error']:.6g}" for r in checks["grid_convergence"]) + ", showing decreasing discretization error. Checkpoint reloads are compared with committed predictions. Unit tests also verify weak-form signs, direct derivatives, data isolation, cache-request provenance, benchmark completeness and the saved discovery outcome.")
    paragraph("Reproduce the packaged run with python -m discovery run; inspect saved artifacts with python -m discovery verify; run tests with python -m pytest -q. The README specifies the environment, observation contract and individual-stage commands. The pinned model snapshots and exact recorded request cache make pretrained inference reproducible; the numerical results use fixed seeds and CPU float64 arithmetic.")
    paragraph("The current engine is intentionally limited to its declared periodic scalar grammar. Unavailable mechanisms, poor excitation, interpolation bias and optimization failure can make inference inconclusive. The main initial condition is inferred, while boundary type is provided. Real experimental data, arbitrary geometries, unobserved boundary inference and calibrated distribution-shift guarantees are outside this demonstrated scope.")
    story.append(PageBreak())

    title("References and artifact provenance")
    evidence = read_jsonl(root / "data" / "evidence.jsonl")
    sources = {}
    for record in evidence:
        sources.setdefault(record["source"], record)
    ordered = ["lec_02/", "lec_04/", "lec_05/", "lec_09/", "4-4-the-logistic-equation", "1509.03580", "2007.02848", "1711.10561", "1711.10566", "2109.01050", "2003.06097"]
    for i, suffix in enumerate(ordered, 1):
        record = next(r for url, r in sources.items() if url.endswith(suffix))
        paragraph(f"[{i}] {escape(record['title'])}. {escape(record['location'])}.<br/><link href='{escape(record['source'])}' color='#1f5275'>{escape(record['source'])}</link>", small=True)
    paragraph("Pretrained models: Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Their official model cards describe architecture and use. Model weights are not duplicated in the submission; the cache records actual requests and outputs.", small=True)
    paragraph("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct<br/>https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2", small=True)
    paragraph("The observation data are generated by the supplied simulator, not imported measurements. Evidence text is originally paraphrased, with external sources attributed. The independent numerical methods, code, benchmarks and reporting logic are included in the artifact. Source publications and pretrained weights retain their own terms.", small=True)

    def footer(canvas, document):
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(1.9 * cm, 1.15 * cm, "Scientific Discovery Engine | Executed Research Artifact")
        canvas.drawRightString(19.1 * cm, 1.15 * cm, str(document.page))

    document = SimpleDocTemplate(str(root / "technical_report.pdf"), pagesize=(21 * cm, 29.7 * cm),
                                  rightMargin=1.9 * cm, leftMargin=1.9 * cm, topMargin=1.65 * cm, bottomMargin=1.7 * cm,
                                  title="Generative Scientific Discovery Engine", author="")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
