

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(craves c b)
(craves d a)
(planet e)
(province c)
(province d)
(province e)
)
(:goal
(and
(craves a c)
(craves b a)
(craves d e)
(craves e b))
)
)


