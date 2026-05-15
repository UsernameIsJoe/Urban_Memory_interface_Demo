# Timeline Preview

Interactive viewer for CLIPSeg study outputs. Hover over the district vibe timeline (left) to preview the corresponding frame-level report and segmentation analytics (right).

This interface was developed as part of the broader **Urban Memory** research project, which investigates how urban environments can be interpreted through layered perceptual signals extracted from image sequences. The tool is intended as a lightweight inspection and navigation interface for rapidly reviewing segmentation outputs across large visual datasets.

---

## Project Context

Urban Memory explores how machine vision and semantic perception models can be used to analyze atmosphere, environmental character, and perceptual continuity within urban space.

The broader workflow combines:

- CLIPSeg-based semantic segmentation
- Prompt–vibe perceptual mapping
- Sequential image analysis
- Temporal urban perception studies
- Frame-level analytic aggregation

Rather than reviewing static outputs individually, this interface enables researchers to move through an entire visual sequence interactively, allowing segmentation behavior and perceptual shifts to be inspected continuously over time.

---

## Interface Features

- Interactive hover-based timeline navigation
- Instant frame preview and report switching
- Fast inspection of CLIPSeg outputs
- Lightweight local viewer
- Designed for large sequential datasets
- Useful for debugging, validation, and comparative analysis

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the interface locally:

```bash
python preview_interface.py
```

Then open the local viewer in your browser.

Hover over the timeline visualization to inspect the matching frame-level segmentation report and analytics.

---

## Purpose Within the Workflow

This tool is not intended as the primary analysis pipeline itself, but as a supporting visualization layer for navigating and validating generated outputs.

It helps:

- Review segmentation consistency across sequences
- Compare perceptual transitions between frames
- Detect unstable or noisy prompt activations
- Rapidly navigate large report collections
- Improve interpretability of model behavior

---

## Research Direction

The Urban Memory project investigates how perceptual qualities of cities may be translated into structured semantic representations using computer vision and sequential analysis methods.

The long-term goal is to better understand how environmental perception, atmosphere, and urban memory emerge through continuous visual experience rather than isolated static scenes.
