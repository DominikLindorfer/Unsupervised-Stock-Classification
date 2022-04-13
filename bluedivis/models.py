from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Dweet(models.Model):

    # user = models.ForeignKey(User,
    #                          related_name="dweets",
    #                          on_delete=models.DO_NOTHING)
    body = models.CharField(max_length=140)
    created_at = models.DateTimeField(auto_now_add=True)
    cluster = None
    ischampion = "test1"
    id = None
    url = "finance.yahoo.com"

    def __str__(self):
        return (
            # f"{self.user} "
            f"({self.created_at:%Y-%m-%d %H:%M}): "
            f"{self.body[:30]}..."
        )
    
    def set_url(self):
        self.url = "https://finance.yahoo.com/quote/" + self.body + "?p=" + self.body + "&.tsrc=fin-srch"

class MLResult(models.Model):

    # "TICKER", "divs_paid", "div_yield_mean", "cluster"
    TICKER = models.TextField()
    divs_paid = models.IntegerField()
    div_yield_mean = models.FloatField()
    cluster = models.IntegerField()

    def __str__(self):
        return (
            f"{self.TICKER}"
            f"{self.divs_paid}"
            f"{self.div_yield_mean}"
            f"{self.cluster}"
        )


