# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:444-505 (Copyright (C) 2016-2026 Brian Bailey).
"""The plant library registry (``initializePlantModelRegistrations``).

Every Helios model is registered, in the C++ order and with its plant type and leaf-inclination Beta,
so ``getAvailablePlantModels()`` matches Helios. Only ``tomato`` has a ported shoot initializer; the
others raise ``NotImplementedError`` when loaded. No builder is ported yet (W3-E).
"""
from __future__ import annotations

from . import tomato

# (name, plant_type, leaf_inclination_Beta) in PlantLibrary.cpp:446-505 order
MODELS = (
    ("almond", "tree", None), ("almond_aldrich", "tree", None), ("almond_independence", "tree", None),
    ("almond_wood_colony", "tree", None), ("apple", "tree", None), ("apple_fruitingwall", "tree", None),
    ("asparagus", "herbaceous", None), ("bindweed", "weed", None), ("bean", "herbaceous", None),
    ("bougainvillea", "herbaceous", None), ("capsicum", "herbaceous", None), ("cheeseweed", "weed", None),
    ("cowpea", "herbaceous", (1.398, 1.574)), ("grapevine_VSP", "herbaceous", None),
    ("grapevine_Wye", "herbaceous", None), ("groundcherryweed", "weed", None), ("maize", "herbaceous", None),
    ("olive", "tree", None), ("pistachio", "tree", (1.32, 1.85)), ("puncturevine", "weed", None),
    ("easternredbud", "tree", (1.00, 2.20)), ("rice", "herbaceous", None), ("butterlettuce", "herbaceous", None),
    ("sorghum", "herbaceous", None), ("soybean", "herbaceous", None), ("strawberry", "herbaceous", None),
    ("sugarbeet", "herbaceous", None), ("tomato", "herbaceous", None), ("cherrytomato", "herbaceous", None),
    ("walnut", "tree", None), ("wheat", "herbaceous", None), ("cucumber", "herbaceous", None),
)

# Ported shoot initializers: name -> fn(plant_architecture)
SHOOT_INITIALIZERS = {"tomato": tomato.initializeTomatoShoots}
# Ported plant builders: name -> fn(plant_architecture, base_position) -> plantID (W3-E)
PLANT_BUILDERS: dict = {}


def _not_ported(what, name):
    def fn(*a, **k):
        raise NotImplementedError(f"{what} for plant model '{name}' is not ported to Heliades yet")
    return fn


def register_all(pa):
    """PlantArchitecture::initializePlantModelRegistrations (structure part; build-parameter
    declarations, PL:507-520, are not ported: tomato declares none)."""
    for name, plant_type, beta in MODELS:
        init = SHOOT_INITIALIZERS.get(name)
        build = PLANT_BUILDERS.get(name)
        pa.registerPlantModel(name,
                              (lambda f=init: f(pa)) if init else _not_ported("shoot initializer", name),
                              (lambda pos, f=build: f(pa, pos)) if build else _not_ported("plant builder", name),
                              plant_type, beta if beta is not None else (0.0, 0.0))


# W4-B: bean and soybean (append-only registration)
from . import bean, soybean  # noqa: E402
SHOOT_INITIALIZERS["bean"] = bean.initializeBeanShoots
SHOOT_INITIALIZERS["soybean"] = soybean.initializeSoybeanShoots
PLANT_BUILDERS["bean"] = bean.buildBeanPlant
PLANT_BUILDERS["soybean"] = soybean.buildSoybeanPlant

# W4-A: cherry tomato and capsicum (shoot initializers and plant builders)
from . import capsicum, cherrytomato  # noqa: E402
SHOOT_INITIALIZERS.update(cherrytomato=cherrytomato.initializeCherryTomatoShoots, capsicum=capsicum.initializeCapsicumShoots)
PLANT_BUILDERS.update(cherrytomato=cherrytomato.buildCherryTomatoPlant, capsicum=capsicum.buildCapsicumPlant)

# W4-C: strawberry and maize (shoot initializer + builder)
from . import maize, strawberry  # noqa: E402
SHOOT_INITIALIZERS["strawberry"] = lambda pa: strawberry.initializeStrawberryShoots(pa)
PLANT_BUILDERS["strawberry"] = lambda pa, pos: strawberry.buildStrawberryPlant(pa, pos)
SHOOT_INITIALIZERS["maize"] = lambda pa: maize.initializeMaizeShoots(pa)
PLANT_BUILDERS["maize"] = lambda pa, pos: maize.buildMaizePlant(pa, pos)

# W4-D: grapevine_VSP (shoot initializer + builder; its build parameters are registered by
# PlantArchitecture.initializePlantModelRegistrations through build_parameters.register_build_parameters)
from . import grapevine  # noqa: E402

SHOOT_INITIALIZERS["grapevine_VSP"] = grapevine.initializeGrapevineVSPShoots
PLANT_BUILDERS["grapevine_VSP"] = grapevine.buildGrapevineVSP

# W5-D: nut trees (almond x4, pistachio, walnut); their build parameters are registered by
# build_parameters.register_build_parameters (trees.register_build_parameters)
from . import almond  # noqa: E402

SHOOT_INITIALIZERS.update(almond=almond.initializeAlmondTreeShoots,
                          almond_aldrich=almond.initializeAlmondTreeAldrichShoots,
                          almond_independence=almond.initializeAlmondTreeIndependenceShoots,
                          almond_wood_colony=almond.initializeAlmondTreeWoodColonyShoots)
PLANT_BUILDERS.update(almond=almond.buildAlmondTree, almond_aldrich=almond.buildAlmondTreeAldrich,
                      almond_independence=almond.buildAlmondTreeIndependence,
                      almond_wood_colony=almond.buildAlmondTreeWoodColony)
from . import walnut  # noqa: E402

SHOOT_INITIALIZERS["walnut"] = walnut.initializeWalnutTreeShoots
PLANT_BUILDERS["walnut"] = walnut.buildWalnutTree
from . import pistachio  # noqa: E402

SHOOT_INITIALIZERS["pistachio"] = pistachio.initializePistachioTreeShoots
PLANT_BUILDERS["pistachio"] = pistachio.buildPistachioTree
# W5-B: herbaceous vegetables (shoot initializers + builders)
from . import asparagus, butterlettuce, cowpea, cucumber, sugarbeet  # noqa: E402

SHOOT_INITIALIZERS.update(asparagus=asparagus.initializeAsparagusShoots,
                          butterlettuce=butterlettuce.initializeButterLettuceShoots,
                          sugarbeet=sugarbeet.initializeSugarbeetShoots, cucumber=cucumber.initializeCucumberShoots,
                          cowpea=cowpea.initializeCowpeaShoots)
PLANT_BUILDERS.update(asparagus=asparagus.buildAsparagusPlant, butterlettuce=butterlettuce.buildButterLettucePlant,
                      sugarbeet=sugarbeet.buildSugarbeetPlant, cucumber=cucumber.buildCucumberPlant,
                      cowpea=cowpea.buildCowpeaPlant)
# W5-C: weeds (bindweed, cheeseweed, groundcherryweed, puncturevine) and bougainvillea
from . import bindweed, bougainvillea, cheeseweed, groundcherryweed, puncturevine  # noqa: E402

SHOOT_INITIALIZERS.update(bindweed=bindweed.initializeBindweedShoots, cheeseweed=cheeseweed.initializeCheeseweedShoots,
                          groundcherryweed=groundcherryweed.initializeGroundCherryWeedShoots,
                          puncturevine=puncturevine.initializePuncturevineShoots,
                          bougainvillea=bougainvillea.initializeBougainvilleaShoots)
PLANT_BUILDERS.update(bindweed=bindweed.buildBindweedPlant, cheeseweed=cheeseweed.buildCheeseweedPlant,
                      groundcherryweed=groundcherryweed.buildGroundCherryWeedPlant,
                      puncturevine=puncturevine.buildPuncturevinePlant,
                      bougainvillea=bougainvillea.buildBougainvilleaPlant)

# W5-C: grapevine_Wye (its build parameters are registered with grapevine_VSP's, grapevine.register_build_parameters)
from . import grapevine_wye  # noqa: E402

SHOOT_INITIALIZERS["grapevine_Wye"] = grapevine_wye.initializeGrapevineWyeShoots
PLANT_BUILDERS["grapevine_Wye"] = grapevine_wye.buildGrapevineWye

# W5-E: fruit / ornamental trees (apple, apple_fruitingwall, olive, easternredbud); the build parameters of
# apple_fruitingwall and olive are registered by trees.register_build_parameters
from . import apple, easternredbud, olive  # noqa: E402

SHOOT_INITIALIZERS.update(apple=apple.initializeAppleTreeShoots,
                          apple_fruitingwall=apple.initializeAppleFruitingWallShoots,
                          olive=olive.initializeOliveTreeShoots,
                          easternredbud=easternredbud.initializeEasternRedbudShoots)
PLANT_BUILDERS.update(apple=apple.buildAppleTree, apple_fruitingwall=apple.buildAppleFruitingWall,
                      olive=olive.buildOliveTree, easternredbud=easternredbud.buildEasternRedbudPlant)
# W5-A: the library grasses (wheat, rice, sorghum; maize above). Their build parameters are registered by
# build_parameters.register_build_parameters through cereals_registry.
from . import wheat  # noqa: E402

SHOOT_INITIALIZERS["wheat"] = lambda pa: wheat.initializeWheatShoots(pa)
PLANT_BUILDERS["wheat"] = lambda pa, pos: wheat.buildWheatPlant(pa, pos)
from . import sorghum  # noqa: E402

SHOOT_INITIALIZERS["sorghum"] = lambda pa: sorghum.initializeSorghumShoots(pa)
PLANT_BUILDERS["sorghum"] = lambda pa, pos: sorghum.buildSorghumPlant(pa, pos)
from . import rice  # noqa: E402

SHOOT_INITIALIZERS["rice"] = lambda pa: rice.initializeRiceShoots(pa)
PLANT_BUILDERS["rice"] = lambda pa, pos: rice.buildRicePlant(pa, pos)
