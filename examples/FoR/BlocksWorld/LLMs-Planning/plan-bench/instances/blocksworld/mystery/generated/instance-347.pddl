(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f j g)
(:init 
(harmony)
(planet f)
(planet j)
(planet g)
(province f)
(province j)
(province g)
)
(:goal
(and
(craves f j)
(craves j g)
)))