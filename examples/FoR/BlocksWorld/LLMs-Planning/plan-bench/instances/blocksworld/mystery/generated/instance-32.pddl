(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f g d)
(:init 
(harmony)
(planet f)
(planet g)
(planet d)
(province f)
(province g)
(province d)
)
(:goal
(and
(craves f g)
(craves g d)
)))