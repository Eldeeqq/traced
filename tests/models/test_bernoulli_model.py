from traced import BernoulliModel


class TestBernoulliModel:
    """Test the BernoulliModel class."""

    def test_init(self):
        model = BernoulliModel("u", "v")
        assert model.success_prob == 0.5
        assert model.success_var == 0.25

    def test_update(self):
        model = BernoulliModel("u", "v")
        model.log(123, True)
        assert model.success_prob == 1.0
        assert model.success_var == 0.0

        model.log(123, False)
        assert model.success_prob == 0.5
        assert 84e-3 - model.success_var < 1e-3

        model.log(124, True)
        model.log(125, True)
        model.log(126, True)
        model.log(127, True)
        model.log(128, True)
        model.log(129, True)
        model.log(130, True)
        model.log(131, True)
        model.log(132, True)

        assert abs(model.success_prob - 10 / 11) < 1e-5
        assert abs(model.success_var) < 7e-3

    def test_to_frame(self):
        model = BernoulliModel("u", "v")

        model.log(124, True)
        model.log(125, True)
        model.log(126, True)
        model.log(127, False)
        model.log(128, False)
        model.log(129, False)
        model.log(130, False)
        model.log(131, True)
        model.log(132, True)

        df = model.to_frame()
        assert df.shape == (9, 7)

    def test_plot(self):
        model = BernoulliModel("u", "v")

        model.log(124, True)
        model.log(125, True)
        model.log(126, True)
        model.log(127, False)
        model.log(128, False)
        model.log(129, False)
        model.log(130, False)
        model.log(131, True)
        model.log(132, True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        model.plot(ax)
        plt.close(fig)


    def test_get_data(self):
        model = BernoulliModel("u", "v")
        model.log(123, True)
        data = model.get_data()
        assert "successes" in data
        assert "failures" in data
        assert "success_prob" in data
        assert "success_var" in data