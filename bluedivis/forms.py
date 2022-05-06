# dwitter/forms.py

from django import forms
from .models import Dweet

# class DweetForm(forms.ModelForm):
#     body = forms.CharField(required=True)

#     class Meta:
#         model = Dweet
#         exclude = ("user", )

class DweetForm(forms.ModelForm):
    body = forms.CharField(
        required=True,
        widget=forms.widgets.Textarea(
            attrs={
                "placeholder": "Enter a TICKER...",
                "class": "input is-link is-large",
            }
        ),
        label="",
    )

    class Meta:
        model = Dweet
        exclude = ("user","is_champion", "is_bluechip",)