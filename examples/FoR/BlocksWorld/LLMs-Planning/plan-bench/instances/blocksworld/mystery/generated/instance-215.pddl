(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l c a f)
(:init 
(harmony)
(planet l)
(planet c)
(planet a)
(planet f)
(province l)
(province c)
(province a)
(province f)
)
(:goal
(and
(craves l c)
(craves c a)
(craves a f)
)))