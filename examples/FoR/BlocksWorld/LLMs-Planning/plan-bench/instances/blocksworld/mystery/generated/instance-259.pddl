(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i h f)
(:init 
(harmony)
(planet i)
(planet h)
(planet f)
(province i)
(province h)
(province f)
)
(:goal
(and
(craves i h)
(craves h f)
)))