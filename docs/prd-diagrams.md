# PRD Diagrams

These diagrams are derived from the canonical PRD in [docs/prd.md](/Users/thebkht/Projects/smart-parking-system/docs/prd.md).

## YOLO Two-Stage Spot Classification

This diagram mirrors the style of the reference image, but matches the actual PRD architecture: quadrilateral pooling followed by per-patch `YOLOv8-cls` inference.

```mermaid
flowchart LR
    I["Parking Lot Image<br/>4000 x 3000"] --> R["Annotated Parking Space Quadrilaterals"]
    R --> W["Perspective Warp<br/>order_corners() + warpPerspective<br/>128 x 128 per spot"]

    W --> P1["Spot Patch A"]
    W --> P2["Spot Patch B"]
    W --> P3["Spot Patch C"]
    W --> P4["Spot Patch D"]

    P1 --> C1["YOLOv8-cls"]
    P2 --> C2["YOLOv8-cls"]
    P3 --> C3["YOLOv8-cls"]
    P4 --> C4["YOLOv8-cls"]

    C1 --> S1["occupied / free<br/>confidence"]
    C2 --> S2["occupied / free<br/>confidence"]
    C3 --> S3["occupied / free<br/>confidence"]
    C4 --> S4["occupied / free<br/>confidence"]

    S1 --> T["Temporal Smoothing"]
    S2 --> T
    S3 --> T
    S4 --> T

    T --> J["JSON Payload<br/>spots + confidence + timestamp"]
    J --> B["FastAPI Backend"]

    classDef input fill:#f3f4f6,stroke:#4b5563,color:#111827;
    classDef stage fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef patch fill:#cffafe,stroke:#0891b2,color:#111827;
    classDef model fill:#ede9fe,stroke:#7c3aed,color:#111827;
    classDef output fill:#fce7f3,stroke:#db2777,color:#111827;
    classDef backend fill:#dcfce7,stroke:#16a34a,color:#111827;

    class I input;
    class R,W,T stage;
    class P1,P2,P3,P4 patch;
    class C1,C2,C3,C4 model;
    class S1,S2,S3,S4,J output;
    class B backend;
```

## Product System Overview

This diagram summarizes the three PRD flows: owner setup, edge inference, and driver-facing `Find My Car`.

```mermaid
flowchart LR
    subgraph Owner["Owner Setup Flow"]
        O1["4-5 Lot Photos"] --> O2["Layout AI<br/>SfM + BEV"]
        O2 --> O3["2D Map JSON"]
        O2 --> O4["Spot Quadrilateral Polygons"]
    end

    O3 --> DB["SQLite + FastAPI"]
    O4 --> DB

    subgraph Edge["Edge Inference Flow"]
        E1["Live Camera Frame"] --> E2["Load Spot Polygons"]
        E2 --> E3["Quadrilateral Pooling"]
        E3 --> E4["YOLOv8n-cls"]
        E4 --> E5["Temporal Smoothing"]
        E5 --> E6["Occupancy JSON"]
    end

    E6 --> DB

    subgraph Driver["Driver App / Find My Car"]
        D1["Driver Query Photo"] --> D2["POST /park"]
        D2 --> D3["SIFT or MobileNetV3 Match"]
        D3 --> D4["spot_id"]
        D4 --> D5["GET /find/{id}"]
        D5 --> D6["Map Highlight in App"]
    end

    DB --> D2
    DB --> D5
    DB --> M["GET /map + GET /status"]
    M --> UI["Web / Mobile Occupancy Map"]

    classDef owner fill:#fef3c7,stroke:#d97706,color:#111827;
    classDef edge fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef driver fill:#fce7f3,stroke:#db2777,color:#111827;
    classDef shared fill:#dcfce7,stroke:#16a34a,color:#111827;

    class O1,O2,O3,O4 owner;
    class E1,E2,E3,E4,E5,E6 edge;
    class D1,D2,D3,D4,D5,D6,UI driver;
    class DB,M shared;
```

## v3 vs v6 Comparison

This side-by-side diagram shows how the project definition moved from the earlier static-camera, ROI-based flow to the current ACPDS-driven quadrilateral pooling and app-integrated system.

```mermaid
flowchart TB
    subgraph V3["v3 Architecture"]
        V3I["Static Camera Frame"] --> V3R["Fixed ROI Boxes<br/>config.yaml"]
        V3R --> V3C["Rectangular Crop per Spot"]
        V3C --> V3M["YOLOv8-cls"]
        V3M --> V3S["Temporal Smoothing"]
        V3S --> V3J["JSON to FastAPI"]

        V3N1["Stage 1 story"] --> V3N2["Deployment defaults to fixed ROIs"]
        V3N3["Product scope"] --> V3N4["Occupancy detection only"]
        V3N5["Dataset story"] --> V3N6["PKLot / CNRPark / static demo baseline"]
    end

    subgraph V6["v6 Architecture"]
        V6I["Parking Lot Image / Live Frame"] --> V6R["ACPDS or SfM Spot Quadrilaterals"]
        V6R --> V6W["order_corners() + warpPerspective<br/>128 x 128 patch per spot"]
        V6W --> V6M["YOLOv8n-cls / s-cls / m-cls"]
        V6M --> V6S["Temporal Smoothing"]
        V6S --> V6J["JSON to FastAPI"]
        V6J --> V6A["Web / Mobile App"]
        V6A --> V6F["Find My Car"]

        V6N1["Stage 1 story"] --> V6N2["Quadrilateral pooling from ACPDS or SfM layouts"]
        V6N3["Product scope"] --> V6N4["Occupancy + owner setup + app + Find My Car"]
        V6N5["Dataset story"] --> V6N6["ACPDS with unseen-lot evaluation"]
    end

    classDef v3 fill:#f3f4f6,stroke:#6b7280,color:#111827;
    classDef v6 fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef v3note fill:#fef3c7,stroke:#d97706,color:#111827;
    classDef v6note fill:#dcfce7,stroke:#16a34a,color:#111827;

    class V3I,V3R,V3C,V3M,V3S,V3J v3;
    class V6I,V6R,V6W,V6M,V6S,V6J,V6A,V6F v6;
    class V3N1,V3N2,V3N3,V3N4,V3N5,V3N6 v3note;
    class V6N1,V6N2,V6N3,V6N4,V6N5,V6N6 v6note;
```

## Key Differences

- `v3` uses fixed ROI boxes and rectangular crops for a mostly static-camera occupancy pipeline.
- `v6` uses ACPDS-style quadrilateral pooling with `order_corners()` and `warpPerspective` before `YOLOv8-cls`.
- `v3` is mainly an edge inference story; `v6` expands the product to owner setup, app delivery, and `Find My Car`.
- `v6` also changes the evaluation story to ACPDS unseen-lot generalization rather than a static demo-first framing.

## Optional Slide Caption

Use this if you want a short caption under the first figure:

> Smart Parking System uses ACPDS quadrilateral pooling to extract one perspective-corrected `128 x 128` patch per parking space, then classifies each patch with `YOLOv8-cls` before temporal smoothing and backend update.
