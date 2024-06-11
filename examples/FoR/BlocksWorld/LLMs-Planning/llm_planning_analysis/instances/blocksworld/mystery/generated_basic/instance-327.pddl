

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b a)
(craves c b)
(planet d)
(province c)
(province d)
)
(:goal
(and
(craves a b))
)
)


