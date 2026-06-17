"""
finder/forms.py — Django form for the user eligibility profile.
Mirrors the CLI intake from step4_profile_collector but as an HTML form.
"""

from django import forms

STATES = [
    ("", "— Select your state —"),
    ("Andhra Pradesh", "Andhra Pradesh"),
    ("Arunachal Pradesh", "Arunachal Pradesh"),
    ("Assam", "Assam"),
    ("Bihar", "Bihar"),
    ("Chhattisgarh", "Chhattisgarh"),
    ("Delhi", "Delhi"),
    ("Goa", "Goa"),
    ("Gujarat", "Gujarat"),
    ("Haryana", "Haryana"),
    ("Himachal Pradesh", "Himachal Pradesh"),
    ("Jharkhand", "Jharkhand"),
    ("Karnataka", "Karnataka"),
    ("Kerala", "Kerala"),
    ("Madhya Pradesh", "Madhya Pradesh"),
    ("Maharashtra", "Maharashtra"),
    ("Manipur", "Manipur"),
    ("Meghalaya", "Meghalaya"),
    ("Mizoram", "Mizoram"),
    ("Nagaland", "Nagaland"),
    ("Odisha", "Odisha"),
    ("Punjab", "Punjab"),
    ("Rajasthan", "Rajasthan"),
    ("Sikkim", "Sikkim"),
    ("Tamil Nadu", "Tamil Nadu"),
    ("Telangana", "Telangana"),
    ("Tripura", "Tripura"),
    ("Uttar Pradesh", "Uttar Pradesh"),
    ("Uttarakhand", "Uttarakhand"),
    ("West Bengal", "West Bengal"),
    ("Jammu and Kashmir", "Jammu and Kashmir"),
    ("Ladakh", "Ladakh"),
    ("Chandigarh", "Chandigarh"),
    ("Puducherry", "Puducherry"),
]

GENDER_CHOICES = [
    ("", "— Select gender —"),
    ("male", "Male"),
    ("female", "Female"),
    ("other", "Other / Prefer not to say"),
]

CASTE_CHOICES = [
    ("", "— Select category —"),
    ("General", "General"),
    ("OBC", "OBC (Other Backward Class)"),
    ("SC", "SC (Scheduled Caste)"),
    ("ST", "ST (Scheduled Tribe)"),
    ("Minority", "Minority"),
]

OCCUPATION_CHOICES = [
    ("", "— Select occupation —"),
    ("farmer", "Farmer / Agricultural Worker"),
    ("student", "Student"),
    ("woman entrepreneur", "Woman Entrepreneur / Self-Employed"),
    ("senior citizen", "Senior Citizen"),
    ("differently abled", "Differently Abled (Divyang)"),
    ("worker", "Worker / Labourer"),
    ("other", "Other"),
]


class EligibilityForm(forms.Form):
    state = forms.ChoiceField(
        choices=STATES,
        widget=forms.Select(attrs={"class": "form-select form-select-lg", "id": "id_state"}),
        label="State / Union Territory",
    )
    age = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(
            attrs={"class": "form-control form-control-lg", "placeholder": "e.g. 28", "id": "id_age"}
        ),
        label="Your Age",
    )
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={"class": "form-select form-select-lg", "id": "id_gender"}),
        label="Gender",
    )
    annual_income = forms.IntegerField(
        min_value=0,
        max_value=100_000_000,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control form-control-lg",
                "placeholder": "e.g. 120000",
                "id": "id_annual_income",
            }
        ),
        label="Annual Family Income (₹)",
        help_text="Enter the total annual income of your household in Indian Rupees.",
    )
    caste_category = forms.ChoiceField(
        choices=CASTE_CHOICES,
        widget=forms.Select(attrs={"class": "form-select form-select-lg", "id": "id_caste_category"}),
        label="Caste / Social Category",
    )
    occupation_type = forms.ChoiceField(
        choices=OCCUPATION_CHOICES,
        widget=forms.Select(attrs={"class": "form-select form-select-lg", "id": "id_occupation_type"}),
        label="Primary Occupation",
    )

    def clean_state(self):
        val = self.cleaned_data["state"]
        if not val:
            raise forms.ValidationError("Please select your state.")
        return val

    def clean_gender(self):
        val = self.cleaned_data["gender"]
        if not val:
            raise forms.ValidationError("Please select your gender.")
        return val

    def clean_caste_category(self):
        val = self.cleaned_data["caste_category"]
        if not val:
            raise forms.ValidationError("Please select your caste category.")
        return val

    def clean_occupation_type(self):
        val = self.cleaned_data["occupation_type"]
        if not val:
            raise forms.ValidationError("Please select your occupation.")
        return val

    def to_profile_dict(self) -> dict:
        """Return a dict compatible with Step 3's retrieve_matching_schemes()."""
        data = self.cleaned_data
        return {
            "state": data["state"],
            "age": data["age"],
            "gender": data["gender"],
            "annual_income": data["annual_income"],
            "caste_category": data["caste_category"],
            "occupation_type": data["occupation_type"],
        }
