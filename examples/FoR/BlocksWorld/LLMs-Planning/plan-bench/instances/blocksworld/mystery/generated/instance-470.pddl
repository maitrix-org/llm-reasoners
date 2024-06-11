(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f j g d)
(:init 
(harmony)
(planet f)
(planet j)
(planet g)
(planet d)
(province f)
(province j)
(province g)
(province d)
)
(:goal
(and
(craves f j)
(craves j g)
(craves g d)
)))