from generate_darknet_labels import generate_label


def test_generate_label():
    test_cases = [
        {
            'row': {
                'x': '256.0',
                'y': '512.0',
                'width': '512.0',
                'height': '256.0',
                'Target': '1'
            },
            'label': ['0', 0.5, 0.625, 0.5, 0.25],
        },
    ]

    for tc in test_cases:
        assert generate_label(tc['row'], 1024, 1024) == tc['label']
