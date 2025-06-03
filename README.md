# Interactive Visual Performance Exploration

Supplementary material to the ACM TODAES journal article *Interactive Visual Performance Space Exploration of Operational Amplifiers with Differentiable Neural Network Surrogate Models*

:globe_with_meridians: Web-App: [Streamlit](https://ed-pcp-demo.streamlit.app/)

:computer: Source code: [GitHub](https://github.com/electronics-and-drives/pcp-demo)

:book: Article: TBD

## Usage

NOTE: Videos may not work on Apple devices.

**Axes and Export**

[axis.webm](https://github.com/AnonCod3/pcp-demo/assets/140701790/701b1d64-2e87-4421-8f41-a106e152f75f)

**Prosection**

[prosection.webm](https://github.com/AnonCod3/pcp-demo/assets/140701790/c1407e25-2487-4ba2-ae7b-07bc4a817097)

**Distribution**

[distribution.webm](https://github.com/AnonCod3/pcp-demo/assets/140701790/07adbe21-a13e-48f9-b524-cde0e8f37563)

**Confusion**

[confusion.webm](https://github.com/AnonCod3/pcp-demo/assets/140701790/9b89a5c2-cf3d-4ee4-9a5a-550c96378bfe)

**gm/ID**

[gmoverid.webm](https://github.com/AnonCod3/pcp-demo/assets/140701790/d8447479-ace5-4e08-a26b-249c1189358d)

## Run Locally

Adjust `port` in `./.streamlit/config.toml`

```sh
$ streamlit run ./ui.py
```

## Known Issues

- [ ] Changing primary column for coloring reverts color map to turbo
- [ ] Changing ckt model after removing a column through the UI adds design
  paramters to pcp that previously didn't exist, possibly an issue with hiding
  columns in `make_experiment`.
- [ ] Sometimes **Performance Space Exploration** plot disappears after switching tabs.
