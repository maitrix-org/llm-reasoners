(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h f g l)
(:init 
(harmony)
(planet h)
(planet f)
(planet g)
(planet l)
(province h)
(province f)
(province g)
(province l)
)
(:goal
(and
(craves h f)
(craves f g)
(craves g l)
)))