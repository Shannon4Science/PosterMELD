# Common generation controls

Pass PosterMELD controls after the launcher's `--` separator.

## Layout and density

```text
--layout-template auto
--layout-template adaptive_auto
--layout-template cluster_<id>
--visual-density lean|balanced|rich
```

List the complete runtime template library:

```bash
python "$LAUNCHER" templates
```

## Style and background

```text
--poster-style navy_serif|teal_modern|burgundy_classic
--background-style auto|minimal_solid|tech_grid|academic_paper|cartographic|blueprint|geometric_soft
--background-palette auto|light_blue|light_gray|warm_ivory|mint|lavender|rose|amber
```

## Header and identity

```text
--conference AAAI
--logo /absolute/path/to/venue-logo.svg
--aff-logo /absolute/path/to/institution-logo.svg
--affiliation-logo-mode single|multi
--header-route auto|classic_left|centered|right_title|split_logos
--header-title-wrap auto|single_line|two_line
```

Use explicit Logo files when the user provides them. Otherwise keep automatic institution Logo resolution enabled.

## Visual quality

The launcher already enables these options:

```text
--enable-generated-teaser
--enable-generated-background
--enable-vlm-layout-review
--enable-visual-legibility-review
--enable-block-vlm-review
```

Do not disable them when the user requests output comparable to the complete PosterMELD pipeline.
