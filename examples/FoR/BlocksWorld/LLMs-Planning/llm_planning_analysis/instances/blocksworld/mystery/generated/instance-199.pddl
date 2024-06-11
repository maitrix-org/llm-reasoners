(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i b g)
(:init 
(harmony)
(planet i)
(planet b)
(planet g)
(province i)
(province b)
(province g)
)
(:goal
(and
(craves i b)
(craves b g)
)))