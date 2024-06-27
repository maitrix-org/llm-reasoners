

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(on b c)
(ontable c)
(ontable d)
(clear a)
(clear b)
)
(:goal
(and
(on a d)
(on b a)
(on c b))
)
)


