

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(ontable a)
(on b a)
(ontable c)
(clear b)
(clear c)
)
(:goal
(and
(on a c)
(on c b))
)
)


