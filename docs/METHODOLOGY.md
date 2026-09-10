# Scientific methodology

The flux and directional methods follow the validated implementation at commit `10e63b4675a6b6d3f9eb1746bc9ecb6b04bf0ed5`. This module adds ingestion and execution around those methods. A future formal paper/reference citation belongs here when available.

## Transition probabilities

Use an explicit regular grid and a positive elapsed lag Δt. Grid bounds must be consistent with spacing. Bins include their lower boundary and exclude their upper boundary, with geographic longitude equivalence handled before binning.

`non_overlapping` constructs `[t0 + kΔt, t0 + (k+1)Δt]` while both times fit within the observed trajectory duration. `every_observation` tries `[tk, tk + Δt]` for each observation except the last, rejecting unavailable endpoints. The second mode produces overlapping samples; these modes change the statistical sample and must not be treated as equivalent weighting schemes.

Endpoints are exact when observed, otherwise linearly interpolated between bracketing observations. Longitude follows the shortest angular arc; latitude and Cartesian coordinates interpolate linearly. There is no extrapolation. The optional gap limit rejects interpolation across a bracketing interval longer than the configured maximum. It applies to both boundaries of a segment; rejection leaves subsequent non-overlapping anchors unchanged. A sampling-interval mode is reported as a diagnostic and never restricts the requested positive lag.

A segment contributes only if both endpoints are in the grid. With retained counts Cᵢⱼ:

\[
N_i=\sum_j C_{ij},\qquad P_{ij}=C_{ij}/N_i,\qquad \sum_jP_{ij}=1.
\]

Out-of-domain endpoints do not remain in Nᵢ. Stay transitions i→i are retained. Sparse tables store only occupied links, with positive integer counts and probabilities consistent with those counts. Matrix validation runs before scientific analysis and never repairs or renormalizes an input row.

Endpoint diagnostics count **two boundary evaluations per candidate segment**. A shared endpoint is counted twice when used by adjacent segments. Each evaluation is exact, interpolated, missing, or rejected by the gap limit. Segment rejection categories are exclusive, in this order: missing boundary, gap violation, outside domain. Thus candidate segments equal retained segments plus these three rejection counts. `outside_start` and `outside_end` are overlapping diagnostics within the domain-rejection category. An incomplete final non-overlapping interval is not a candidate. Exact endpoints are unaffected by the gap limit.

## Geometry and statistics

Geographic displacement distances and initial bearings use `pyproj.Geod` with the configured ellipsoid. Cartesian displacements are Euclidean. Bearings are degrees clockwise from north/positive y: 0° is +y, 90° is +x. No map projection transforms scientific coordinates.

The matrix represents transitions between cell centres, not the original within-cell displacement. Stay/moving separation defines P_move = Σⱼ≠ᵢ Pᵢⱼ. Moving-conditioned weights are qᵢⱼ = Pᵢⱼ/P_move when P_move is positive. Support uses **moving observation counts**, not probability, occupied-link counts, or total trajectories.

The statistics preserve outgoing/incoming displacement moments, circular harmonics, entropy, direction comparisons, and support diagnostics. A stay-only source has zero all-population flux/directional vectors; its moving-conditioned direction is undefined.

The frozen analysis requires at least one moving transition somewhere in the matrix. A completely stay-only matrix can be built and saved, but downstream analysis raises a clear error. This existing requirement is retained rather than introducing new scientific behavior for that degenerate case.

## Lagrangian displacement flux pathway

\[
\mathbf U_{\mathrm{out,all}}=
\frac{1}{\Delta t}\sum_{j\ne i}P_{ij}\Delta\mathbf r_{ij},
\qquad [U]=L/T.
\]

We call this vector **Lagrangian displacement flux**, shortened to **flux**. It weights connectivity by physical displacement and expresses an average displacement rate over the full source population, including stays. Rates use `geometry.length_unit/matrix.time_unit`. The scalar `S_flux_rate` used for flux cores is its magnitude; optional support-aware 3×3 smoothing preserves the existing implementation. The `U_*` names continue to denote the same displacement-rate vectors.

The terminology follows the oceanographic convention in which volume flux per unit area has dimensions of velocity, while volume transport through a finite section has dimensions of volume/time; 1 Sv = 10⁶ m³/s. [University of Washington: Flux and Transport](https://uw.pressbooks.pub/ocean285/chapter/flux-and-transport/). Here the estimate comes from row-normalized trajectory transitions between cell centres. It does not establish a volume-weighted ocean current or a measured volume transport: the calculation has no represented water-volume weights or section-area integral.

Supported cells must exceed the configured global flux quantile and pass the existing transverse-ridge comparison. Transverse samples use the mean flux direction and local effective cell scales. The boundary-aware test preserves distinctions between observed flanks, domain boundaries, and missing support. Existing component and segment extraction organizes selected cells without identifying named currents or classifying topology.

Front detection samples flux projected along the core direction across local sections. It retains the existing local core refinement, contiguous median smoothing, along-segment composites, outward-drop selection, and persistence rules. Segment-context detections are reduced to one result per physical core side. Outputs distinguish probable fronts, observable sides without a retained front, and unobservable sides. These are local probable-front diagnostics, not constructed continuous front lines.

### Integrated ridge strength

The optional component/segment diagnostic `integrated_ridge_strength_area_rate` is the trapezoidal line integral of the selected scalar field `S0_field_rate` **along the ridge**:

\[
I_{\mathrm{ridge}}=\sum_{(a,b)\in\mathrm{ridge\ edges}}
\frac{S_{0,a}+S_{0,b}}{2}\,\ell_{ab},\qquad [I_{\mathrm{ridge}}]=L^2/T.
\]

The selected field is raw or smoothed according to `flux.ridge_field`. Closed segments include the closing edge, isolated cells contribute zero, and component values sum their segment integrals. Public columns use the geometry/time suffix, for example `integrated_ridge_strength_km2_day`. This diagnostic combines ridge intensity and length; it is not flux through a cross-section. `rank_integrated_ridge_strength` ranks that integral, while `rank_median_flux` ranks the median raw flux magnitude. The integral and its rankings retain their original numerical definitions.

## Directional pathway

\[
\mathbf D_{\mathrm{out,move}}=\sum_{j\ne i}q_{ij}\hat{\mathbf r}_{ij},
\qquad |\mathbf D_{\mathrm{out,move}}|=R_1,
\]
\[
\mathbf D_{\mathrm{out,all}}=P_{\mathrm{move}}\mathbf D_{\mathrm{out,move}},
\qquad D=|\mathbf D_{\mathrm{out,all}}|=P_{\mathrm{move}}R_1.
\]

R₁ measures first-harmonic directional coherence between 0 and 1; θ₁ is its bearing when defined. Unit displacement directions remove distance weighting. Supported candidate cells simultaneously satisfy absolute thresholds on P_move, R₁, and D. There is **no global directional percentile**.

The candidate area is reduced by a transverse ridge test on D, sampled across θ₁. Approximately equal adjacent raw ridges that pass the existing transverse-step criterion form plateau components. `ridge_plateau_tolerance` bounds the pairwise D difference along these connections; it is not a maximum range across the whole transitive component. Each plateau retains a spatial medoid, minimizing summed physical distances. Near-tied medoids use greater moving support, then lower cell ID. The fixed medoid comparison tolerance is part of the frozen method.

After thinning, local connectivity requires compatible neighboring directions and compatible cell-step axes at both endpoints. Components below `minimum_component_cells` are removed. Directional fronts use the retained directional cores and the existing projected-directional-strength sections, refinement, median profiles, graph-neighborhood composites and persistence selection. Their strengths and losses are dimensionless; their locations and distances use the selected geometry.

## Comparison and validation

Flux describes displacement-weighted movement; directional structure describes distance-free organization. Their overlap is informative, but neither pathway validates the other. Cell and component comparison tables summarize overlap without matching unlike component geometries.

Optional validation compares selected flux fronts to independently calculated spatial gradients and local background statistics. It does not move, accept, or reject a front. Missing support remains missing. An otherwise valid matrix with no cores produces empty core/front outputs; optional gradient validation records that no flux sections were available.
