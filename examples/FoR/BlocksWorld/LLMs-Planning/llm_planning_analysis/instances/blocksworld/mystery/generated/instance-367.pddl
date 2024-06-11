(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c e l)
(:init 
(harmony)
(planet c)
(planet e)
(planet l)
(province c)
(province e)
(province l)
)
(:goal
(and
(craves c e)
(craves e l)
)))