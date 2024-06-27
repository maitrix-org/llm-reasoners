

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(craves c d)
(planet d)
(craves e b)
(province a)
(province c)
(province e)
)
(:goal
(and
(craves a e)
(craves c b)
(craves d a))
)
)


