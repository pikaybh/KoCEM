from schemas.kocem import (KOCEM_FEATURES, 
                            DOMAIN_RESONING_FEATURES, 
                            DRAWING_INTERPRETATION_FEATURES,
                            STANDARD_NOMENCLATURE_FEATURES)


def call_features(subset: str):
    if subset == "Domain_Reasoning":
        return DOMAIN_RESONING_FEATURES
    if subset == "Drawing_Interpretation":
        return DRAWING_INTERPRETATION_FEATURES
    if subset == "Standard_Nomenclature":
        return STANDARD_NOMENCLATURE_FEATURES
    return KOCEM_FEATURES


__all__ = ["call_features"]