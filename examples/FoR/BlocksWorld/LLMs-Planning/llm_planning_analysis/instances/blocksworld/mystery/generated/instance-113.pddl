(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g b f)
(:init 
(harmony)
(planet g)
(planet b)
(planet f)
(province g)
(province b)
(province f)
)
(:goal
(and
(craves g b)
(craves b f)
)))