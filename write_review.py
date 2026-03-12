import os

filepath = r'E:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\docs\sapflow_methods_review.md'

content = []
content.append("# Thermometric Sap Flow Measurement Methods: A Technical Review")
content.append("")
content.append("## 1. Introduction")
content.append("")
content.append("Thermometric methods dominate contemporary sap flow measurement because they are minimally invasive, relatively inexpensive, and amenable to continuous automated logging (Smith & Allen, 1996; Vandegehuchte & Steppe, 2013). All five major approaches exploit the relationship between heat transport in the xylem and sap movement, yet they differ fundamentally in heating geometry, measurement principle, and the data-processing pipeline required to convert raw signals into ecophysiologically meaningful fluxes.")
content.append("")
content.append("### 1.1 Historical Context")
content.append("")
content.append("The foundation of heat-based sap flow measurement traces to Marshall (1958), who formalized heat pulse theory for porous media. Granier (1985) introduced the thermal dissipation probe (TDP), which became the most widely deployed method due to its simplicity. In parallel, the compensation heat pulse velocity (CHPV) method was refined by Green and Clothier (1988) and Swanson and Whitfield (1981), while the heat ratio method (HRM) was developed by Burgess et al. (2001) specifically to capture low and reverse flows. Nadezhdina et al. (1998) proposed the heat field deformation (HFD) approach for radial profiling, and Čermák et al. (1973; 2004) pioneered the trunk heat balance (THB) method for whole-stem energy accounting.")
content.append("")
content.append("### 1.2 Terminology")
content.append("")
content.append("| Term | Symbol | Units | Definition |")
content.append("|------|--------|-------|------------|")
content.append("| Sap flux density | *J*_s or *F*_d | g cm⁻² h⁻¹ or kg m⁻² s⁻¹ | Mass flow rate per unit sapwood area |")
content.append("| Sap velocity | *v*_s | cm h⁻¹ or m s⁻¹ | Linear speed of sap movement |")
content.append("| Sap flow | *Q* | kg h⁻¹ or L h⁻¹ | Volumetric or mass flow per stem |")
content.append("| Heat pulse velocity | *v*_h | cm s⁻¹ | Measured velocity of a heat pulse |")
content.append("| Sapwood area | *A*_sw | cm² or m² | Cross-sectional area of functional xylem |")
content.append("| Zero-flow baseline | *ΔT*_max or *ΔT*_0 | °C | Reference temperature difference under no-flow conditions |")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(content))
print(f'Phase 1 written: {len(content)} lines')
