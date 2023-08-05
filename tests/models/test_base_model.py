from traced import BaseModel


class TestBaseModel:
    """Test the BaseModel class."""

    def test_init(self):
        """Test the constructor."""

        model = BaseModel("u", "v")

        assert model.u == "u"
        assert model.v == "v"
        assert model.n == 0

    def test_log(self):
        """Test the log method."""

        model = BaseModel("u", "v")

        model.log(123)

        assert model.ts == 123
        assert model.n == 1

        model.log(456)
        assert model.ts == 456
        assert model.n == 2
