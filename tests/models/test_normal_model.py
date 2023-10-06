from traced import NormalModel


class TestNormalModel:
    """Test the NormalModel class."""

    def test_init(self):
        model = NormalModel("u", "v")
        assert model.mu == 5.0
        assert model.sigma == 2.0
        assert model.alpha == 1.0
        assert model.beta == 1.0

    def test_update(self):
        model = NormalModel("u", "v", mu_0=10, sigma_0=1)
        model.log(123, 15)
        model.log(124, 14)
        model.log(125, 13)
        model.log(126, 12)
        model.log(127, 11)
        model.log(128, 10)
        assert model.mu == 12.5
        assert abs(2.46 - model.sigma) < 6e-3

    def test_to_frame(self):
        model = NormalModel("u", "v")

        model.log(124, 15)
        model.log(125, 15)
        model.log(126, 15)
        model.log(127, 12)
        model.log(128, 12)
        model.log(129, 12)
        model.log(130, 12)
        model.log(131, 15)
        model.log(132, 15)

        df = model.to_frame()
        assert df.shape[0] == 9

    def test_plot(self):
        model = NormalModel("u", "v")

        model.log(124, 15)
        model.log(125, 15)
        model.log(126, 15)
        model.log(127, 12)
        model.log(128, 12)
        model.log(129, 12)
        model.log(130, 12)
        model.log(131, 110005)
        model.log(132, 15)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # type: ignore
        model.plot(ax)
        fig.savefig("test.png")
        plt.close(fig)

    def test_get_data(self):
        model = NormalModel("u", "v")
        model.log(123, 10)
        model.log(124, 10)
        model.log(125, 13)
        model.log(126, 10)
        model.log(127, 10)
        data = model.get_data()
        assert "observed" in data
        assert "ts" in data
        assert "mu" in data
        assert "sigma" in data
        assert "upper_bound" in data
        assert "lower_bound" in data
