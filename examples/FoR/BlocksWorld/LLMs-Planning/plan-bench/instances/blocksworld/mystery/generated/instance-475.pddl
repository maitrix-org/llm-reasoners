(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g k h e)
(:init 
(harmony)
(planet g)
(planet k)
(planet h)
(planet e)
(province g)
(province k)
(province h)
(province e)
)
(:goal
(and
(craves g k)
(craves k h)
(craves h e)
)))